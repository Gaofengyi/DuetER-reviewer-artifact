"""Immutable real AES-GCM payload/join/document records for timed retrieval.

Preparation uses fresh independent keys and monotonically unique nonces.
Client keys are never passed to the server process. This is a measurement
wrapper over the legacy indices, not a claim of a hardened deployment.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CAPSULE = struct.Struct("!QQI")
RECORD = struct.Struct("!16sII")
U32 = struct.Struct("!I")
DUMMY_ROW = 2 ** 64 - 1


def digest_label(key: bytes, domain: bytes, value: bytes, size=16) -> bytes:
    return hmac.new(key, domain + b"\0" + value, hashlib.sha256).digest()[:size]


class Owner:
    def __init__(self, keys: dict[str, str]):
        self.keys = {key: bytes.fromhex(value) for key, value in keys.items()}
        self.aes = {key: AESGCM(self.keys[key]) for key in ("payload", "join", "document")}
        self.counter = 0
        self.nonce_prefix = os.urandom(4)

    @classmethod
    def fresh(cls):
        return cls({key: os.urandom(32).hex() for key in
                    ("payload", "join", "document", "alias", "locator")})

    def exported(self):
        return {key: value.hex() for key, value in self.keys.items()}

    def protect(self, domain: str, plaintext: bytes, aad: bytes) -> bytes:
        self.counter += 1
        nonce = self.nonce_prefix + self.counter.to_bytes(8, "big")
        return nonce + self.aes[domain].encrypt(nonce, plaintext, aad)

    def open(self, domain: str, protected: bytes, aad: bytes) -> bytes:
        return self.aes[domain].decrypt(protected[:12], protected[12:], aad)

    def locator(self, row: int) -> bytes:
        return digest_label(self.keys["locator"], b"document", row.to_bytes(8, "big"), 8)

    def make_record(self, path: int, cell: int, row: int, payload: bytes) -> bytes:
        context = struct.pack("!BIQ", path, cell, row)
        alias = digest_label(self.keys["alias"], b"path-cell-row", context)
        locator = self.locator(row)
        capsule = CAPSULE.pack(row, int.from_bytes(locator, "big"), 0 if row == DUMMY_ROW else 1)
        encrypted_payload = self.protect("payload", payload, alias)
        encrypted_join = self.protect("join", capsule, alias)
        return (RECORD.pack(alias, len(encrypted_payload), len(encrypted_join))
                + encrypted_payload + encrypted_join)

    def open_record(self, record: bytes):
        alias, payload_length, join_length = RECORD.unpack_from(record)
        if len(record) != RECORD.size + payload_length + join_length:
            raise ValueError("record size mismatch")
        pos = RECORD.size
        payload = self.open("payload", record[pos:pos + payload_length], alias)
        capsule = self.open("join", record[pos + payload_length:], alias)
        row, locator, version = CAPSULE.unpack(capsule)
        if (row == DUMMY_ROW) != (version == 0):
            raise ValueError("invalid dummy capsule")
        return row, locator.to_bytes(8, "big"), version, payload


def pad_lexical_payload(tokens: list[str], token_ids: dict[str, int], bucket=256) -> bytes:
    # Owner-assigned IDs are injective over the prepared token vocabulary;
    # unlike truncating HMAC to uint32, this does not silently collide.
    ids = [token_ids[token] for token in tokens]
    plaintext = U32.pack(len(ids)) + b"".join(U32.pack(value) for value in ids)
    return plaintext + bytes((-len(plaintext)) % bucket)


def open_lexical_payload(payload: bytes) -> list[int]:
    count = U32.unpack_from(payload)[0]
    if 4 + count * 4 > len(payload):
        raise ValueError("invalid lexical token length")
    return [value[0] for value in struct.iter_unpack("!I", payload[4:4 + 4 * count])]


class StoreWriter:
    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite prepared store: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE records(path INTEGER, cell INTEGER, row INTEGER, value BLOB,
                             PRIMARY KEY(path,cell,row)) WITHOUT ROWID;
        CREATE TABLE documents(locator BLOB PRIMARY KEY, value BLOB) WITHOUT ROWID;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;
        """)
        self.payload_bytes = 0
        self.document_bytes = 0

    def record(self, path: int, cell: int, row: int, value: bytes):
        self.db.execute("INSERT INTO records VALUES (?,?,?,?)", (path, cell, row, value))
        self.payload_bytes += len(value)

    def document(self, locator: bytes, value: bytes):
        self.db.execute("INSERT INTO documents VALUES (?,?)", (locator, value))
        self.document_bytes += len(value)

    def finish(self, metadata: dict):
        self.db.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(metadata),))
        self.db.commit()
        self.db.close()


class StoreReader:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.db.execute("PRAGMA query_only=ON")
        self.manifest = json.loads(self.db.execute(
            "SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])

    def record(self, path: int, cell: int, row: int) -> bytes:
        # -1 is SQLite's stored sentinel for the uint64 dummy row.
        found = self.db.execute("SELECT value FROM records WHERE path=? AND cell=? AND row=?",
                                (path, cell, row)).fetchone()
        if found is None:
            raise KeyError(f"missing prepared record path={path} cell={cell} row={row}")
        return found[0]

    def document(self, locator: bytes) -> bytes:
        found = self.db.execute("SELECT value FROM documents WHERE locator=?", (locator,)).fetchone()
        if found is None:
            raise KeyError("missing prepared document locator")
        return found[0]
