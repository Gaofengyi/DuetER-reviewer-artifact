"""Length-framed binary RPC. All byte counts are actual application bytes.

The channel is loopback TCP, without TLS, WAN emulation, or LLM generation.
AEAD protection belongs to stored records; local DPE coordinates are exposed
to the server as required by the evaluated retrieval protocol.
"""
from __future__ import annotations

import json
import socket
import struct
import time

MAGIC = b"DER2"
HEADER = struct.Struct("!4sII")
U64 = struct.Struct("!Q")
MAX_FRAME = 1024 ** 3


def encode(metadata: dict, blocks: list[bytes]) -> bytes:
    meta = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    return b"".join([HEADER.pack(MAGIC, len(meta), len(blocks)), meta]
                    + [U64.pack(len(block)) + block for block in blocks])


def decode(body: bytes) -> tuple[dict, list[bytes]]:
    if len(body) < HEADER.size:
        raise ValueError("truncated RPC header")
    magic, meta_size, count = HEADER.unpack_from(body)
    if magic != MAGIC or meta_size > len(body) - HEADER.size:
        raise ValueError("invalid RPC header")
    pos = HEADER.size
    meta = json.loads(body[pos:pos + meta_size])
    pos += meta_size
    blocks = []
    for _ in range(count):
        if pos + U64.size > len(body):
            raise ValueError("truncated block length")
        size = U64.unpack_from(body, pos)[0]
        pos += U64.size
        if size > len(body) - pos:
            raise ValueError("truncated block")
        blocks.append(body[pos:pos + size])
        pos += size
    if pos != len(body):
        raise ValueError("trailing RPC bytes")
    return meta, blocks


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        block = sock.recv(min(remaining, 1024 * 1024))
        if not block:
            raise EOFError("TCP peer closed the connection")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def send(sock: socket.socket, body: bytes) -> int:
    if len(body) > MAX_FRAME:
        raise ValueError("oversized frame")
    sock.sendall(U64.pack(len(body)) + body)
    return U64.size + len(body)


def receive(sock: socket.socket) -> tuple[bytes, int]:
    size = U64.unpack(recv_exact(sock, U64.size))[0]
    if size > MAX_FRAME:
        raise ValueError("oversized frame")
    return recv_exact(sock, size), U64.size + size


def rpc(sock: socket.socket, meta: dict, blocks: list[bytes]):
    started = time.perf_counter_ns()
    request = encode(meta, blocks)
    encoded = time.perf_counter_ns()
    sent = send(sock, request)
    response, received = receive(sock)
    transferred = time.perf_counter_ns()
    response_meta, response_blocks = decode(response)
    finished = time.perf_counter_ns()
    if response_meta.get("error"):
        raise RuntimeError(response_meta["error"])
    return response_meta, response_blocks, {
        "request_serialize_ms": (encoded - started) / 1e6,
        "transfer_and_server_ms": (transferred - encoded) / 1e6,
        "response_parse_ms": (finished - transferred) / 1e6,
        "rpc_wall_ms": (finished - started) / 1e6,
        "sent_application_bytes": sent,
        "received_application_bytes": received,
    }
