# Support modules required to execute this ablation

The component-ablation runners reuse four small support modules that were
frozen in earlier study directories. They are referenced through the relative
bootstrap paths at the top of `server_adapter.py`, `components_client.py`,
`build_plain_index.py`, `prepare_components.py`, `run_components.py`, and
`proxy_entry.py`:

| Module | Frozen origin | Used for |
|---|---|---|
| `selection.py` (`waterfill`, `local_order`) | `<WORKSPACE>/experiment_projectlift_protocol_20261107/protocol/` | capacity-balanced lexical cell quotas |
| `wire.py` (`encode/decode/send/receive`) | same | TCP framing for the Search/Fetch harness |
| `config.json` (paths, seeds) | same | points at the prepared core caches |
| `records.py` | same | protected-record store |
| `client_runtime_binding.py` + `runtime_binding.json` | `<WORKSPACE>/experiment_pprag_precision_network_20261108/` | pinned CPU runtime binding |
| `client_adapter.py` | same | client-side request execution |
| `network_proxy.py` | same | `proxy_entry.py` network-shaping entry |

`run_components.py` additionally records SHA-256 provenance over
`unified_pipeline.py` (packaged at `experiments/unified/`) and
`project_then_lift_geometry.py` (historical study directory) without importing
them.

To re-execute, place these support modules at the referenced relative
locations (or adjust the bootstrap paths) after preparing the corpora and
caches described in `REPRODUCIBILITY.md`. Reading the ablation logic does not
require them: the registered design is in `EXPERIMENT_PLAN.txt`, and the
paired results are in `results/paper/components/`.
