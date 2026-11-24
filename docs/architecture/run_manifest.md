# Immutable Run Manifest & Separate Result Records (`M3-C`)

**Implementation:** [`src/jssp_research/experiments/manifests.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/experiments/manifests.py), [`src/jssp_research/results/`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/results/), [`src/jssp_research/utils/provenance.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/utils/provenance.py)

---

## 1. Design Principle: Pre-Run Immutability & Outcome Separation

1. **`RunManifest` (`jssp-manifest-v1`):** Built *before* execution starts. All Pydantic sections use `ConfigDict(frozen=True, extra="forbid")`, **and** all nested mappings and sequences (`configuration_snapshot`, `qmc_scramble_metadata`, `algorithm_specific_budgets`, `streams`, `artifact_refs`) are deeply frozen recursively into read-only `MappingProxyType` and `tuple` objects at construction and deserialization (`M3.1 E02`). Caller mutations to original input dicts/lists after manifest creation have zero effect on the manifest, and `manifest.verify_integrity()` re-verifies all stored digests. Contains identity, Git/diff provenance, algorithm configuration snapshot + hash, benchmark instance + raw/canonical SHA-256, fixed makespan objective/decoder metadata, master seed + pairing policy + all 9 derived stream SHA-256 seeds, computational budget, verified host OS/CPU/RAM/toolchain/lockfile provenance, and `canonical_serialization_version = "jssp-canonical-json-v1"`.
2. **Versioned Canonical Serialization Profile (`jssp-canonical-json-v1`, `M3.1 E03`):**
   Adopted **Option A** (documented project-specific canonical JSON serialization profile tied to Python `3.15` + `RunManifest` schema `jssp-manifest-v1`):
   - **Recursive object-key sorting** by Unicode code point (keys restricted to valid UTF-8 `str`).
   - **Strict UTF-8 + explicit string escaping:** Lone UTF-16 surrogates are rejected; `json.dumps(..., separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")` yields deterministic ASCII/UTF-8 bytes with `\uXXXX` escaping.
   - **Exact 64-bit integer precision:** Signed/unsigned integers in `[-(2^63), 2^64 - 1]` (including `u64` values beyond IEEE-754 $2^{53}-1$) are serialized as exact base-10 integer literals without float coercion.
   - **Finite numeric values & `-0.0` normalization:** `NaN`, `+Infinity`, and `-Infinity` are rejected (`ManifestIntegrityError`); IEEE negative zero (`-0.0`) is canonicalized to `+0.0` (`0.0`).
   - **Backward Compatibility:** Pre-M3.1 manifests lacking the top-level `canonical_serialization_version` key validate cleanly under the legacy 8-section `body_dict()` using `jssp-canonical-json-v1` rules without modifying existing manifest hashes; M3.1+ manifests created via `create_run_manifest` explicitly embed `"canonical_serialization_version": "jssp-canonical-json-v1"` inside the hashed body.
3. **`RunResultRecord` (`jssp-result-v1`):** Persisted in *separate* append-only files (`{run_id}.attempt-{attempt_index:03d}.result.json`). Retries create incremented `attempt_index` records without modifying the original manifest or prior attempts. Every result links back via `run_id`, `manifest_sha256`, `experiment_id`, `configuration_id`, `instance_id`, `replicate_index`, and `master_seed`, verified by `validate_result_against_manifest`.

---

## 2. Confirmatory Run Integrity vs. Development Runs

| Requirement | `development` / `pilot` / `screening` | `confirmatory` |
|---|---|---|
| Clean Git working tree | Dirty allowed; **must record** `git_dirty=True` + `git_diff_sha256` | **Mandatory `git_dirty=False` & `git_diff_sha256=None`** (`ConfirmatoryIntegrityError` otherwise) |
| Locked dependencies | Recorded (`Cargo.lock`, `uv.lock` SHA-256 or `None`) | **Mandatory valid 64-char SHA-256** for both `Cargo.lock` and `uv.lock` |
| Master seed schedule | Frozen replicate or explicit `u64` allowed | **Mandatory `master_seed == FROZEN_MASTER_SEEDS_V1[replicate_index]`** (`0..20`) |
| Benchmark track | Track 1–3 allowed | Track 1–3 allowed; deferred Tracks 4/5 **strictly forbidden** |

Unknown hardware fields (e.g., on non-Linux hosts lacking `/proc/cpuinfo` or `/proc/meminfo`) are recorded as `"unavailable"` / `None` — hardware or toolchain values are never guessed.
