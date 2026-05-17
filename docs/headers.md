# `X-InferenceGate-*` Header Contract

InferenceGate exposes a stable, registered header contract that lets callers
(typically [InferenceGlue](../../InferenceGlue)) influence routing, lookup, and
recording without polluting the request body or the cassette content hash.

## Design rules

1. **Strict allowlist.** Any `X-InferenceGate-*` header not in the registry
   below is rejected with HTTP **400** and a diagnostic body listing the
   accepted names. Header matching is case-insensitive.
2. **Stripped before hashing and before upstream forwarding.** None of these
   headers ever participate in the content hash, and none are forwarded to
   the upstream model server.
3. **Reserved namespace.** All path prefix `/gate/*` and all
   `X-InferenceGate-*` header names are owned by Gate; user models must not
   reuse them.

## Header categories

Three suffix categories with rigid discipline:

| Suffix | Phase | Affects hash? | Effect |
|---|---|---|---|
| `Require-*` | Lookup | No | Filter cassettes by metadata; never hits upstream |
| `Metadata-*` | Record | No | Stored in tape frontmatter; ignored on replay |
| `Control-*` | Runtime | No | Per-request mode/strategy override |

## Registry

| Header | Category | Values | Effect |
|---|---|---|---|
| `X-InferenceGate-Require-Engine` | Require | `vllm` / `llama.cpp` / `openai` / free string | Match only cassettes whose `metadata.engine` equals this |
| `X-InferenceGate-Require-Engine-Version` | Require | semver | Match only cassettes with equal `metadata.engine_version` |
| `X-InferenceGate-Require-Fuzzy-Model` | Require | `on` / `off` | Per-request override of session `fuzzy_model` |
| `X-InferenceGate-Require-Fuzzy-Sampling` | Require | `off` / `soft` / `aggressive` | Per-request override of session `fuzzy_sampling` |
| `X-InferenceGate-Require-Exact` | Require | `true` | Forces both fuzzy settings off |
| `X-InferenceGate-Metadata-Engine` | Metadata | free string | Stored in tape `metadata.engine` |
| `X-InferenceGate-Metadata-Engine-Version` | Metadata | free string | Stored as `metadata.engine_version` |
| `X-InferenceGate-Metadata-Test-NodeID` | Metadata | pytest nodeid | Stored as `metadata.test_node_id` |
| `X-InferenceGate-Metadata-Worker-ID` | Metadata | xdist worker / `master` | Stored as `metadata.worker_id` |
| `X-InferenceGate-Metadata-Recorded-By` | Metadata | free string | Stored as `metadata.recorded_by` |
| `X-InferenceGate-Control-Mode` | Control | `replay` / `record` / `passthrough` | Per-request mode override |
| `X-InferenceGate-Control-Reply-Strategy` | Control | `round-robin` / `random` / `first` | Per-request reply selection |

The legacy `X-Gate-Reply-Strategy` is honoured as a deprecated alias for
`X-InferenceGate-Control-Reply-Strategy`.

## Producing headers from clients

### From InferenceGlue

Glue auto-injects context headers on every outbound request via an httpx event
hook. It also consumes the InferenceGate pytest plugin's header context when
Gate is installed. See `inference_glue.request_context`:

```python
from inference_glue.request_context import headers as glue_headers

with glue_headers(**{"X-InferenceGate-Require-Exact": "true"}):
    result = model.complete(...)
```

Per-call `extra_headers=` always wins over context defaults.

### From the pytest plugin

The pytest plugin stores per-test headers in `inference_gate.pytest_context`.
Downstream clients such as InferenceGlue can read that context and attach the
headers to outbound HTTP requests. The marker
`@pytest.mark.inferencegate(fuzzy_model=False, fuzzy_sampling="off")`
translates to `Require-Fuzzy-Model: off` / `Require-Fuzzy-Sampling: off`.
Every test additionally pushes `Metadata-Test-NodeID` and
`Metadata-Worker-ID` automatically.

## Behaviour on mismatch

A REPLAY-mode `Require-*` mismatch produces a **503** "no cached response"
identical to a content-hash miss. This keeps cassette-miss diagnostics
uniform across the matching pipeline.
