# Python–Rust FFI Boundary Specification

**Native Module:** `jssp_research._native` (compiled from `crates/jssp-python` via Maturin + PyO3 `0.30.1`)
**Python Wrapper Package:** `src/jssp_research/native/` (`__init__.py`, `bridge.py`)

---

## 1. Boundary Design Principles

1. **Coarse-Grained Calls:** FFI calls pass whole instances, chromosomes, and schedules (and in later milestones, entire search runs or populations) across the boundary. Inner loops (gap scanning, move evaluation, feasibility validation) execute purely inside Rust without calling Python callbacks.
2. **Rust as Scheduling Source of Truth:** Python does not reimplement OBR decoding, feasibility validation, or makespan calculation for production execution.
3. **Immutable Typed Python Containers:** Wrapper functions accept standard Python `Sequence` inputs and return frozen dataclasses (`ScheduledOperationRecord`, `DecodedScheduleResult`) with full type annotations.
4. **Independent Re-Validation Support:** Both `decode_obr` and `validate_schedule` cross the boundary cleanly, allowing Python property-based tests (Hypothesis) to inspect raw schedules and trigger negative feasibility checks.

---

## 2. Exported FFI Functions (`jssp_research._native` & `jssp_research.native`)

| Python Wrapper Function | Input Arguments | Return Type | Behavior |
|---|---|---|---|
| `get_bridge_info()` / `get_native_bridge_info()` | None | `str` / `dict[str, str]` | Diagnostic crate identity (`jssp-core/0.1.0`) and module version checks. |
| `get_core_version()` | None | `str` | Semantic version of `jssp-core`. |
| `validate_instance(jobs)` | `Sequence[Sequence[tuple[int, int]]]` | `tuple[int, int, int, int]` | Validates rectangular JSSP constraints; returns `(J, M, L, total_work)`. |
| `validate_obr(jobs, chromosome)` | `jobs`, `Sequence[int]` | `bool` | Validates length `J*M`, bounds `0..J`, and exact multiplicity `M`. |
| `resolve_obr_operations(jobs, chromosome)` | `jobs`, `Sequence[int]` | `list[tuple[int, int]]` | Validates OBR and returns resolved `(job, op_index)` pairs. |
| `decode_obr(jobs, chromosome)` | `jobs`, `Sequence[int]` | `DecodedScheduleResult` | Runs canonical Serial SGS earliest idle-gap insertion + independent feasibility check. |
| `decode_obr_append_only(jobs, chromosome)` | `jobs`, `Sequence[int]` | `DecodedScheduleResult` | Runs semi-active append-only reference decoder + feasibility check. |
| `validate_schedule(jobs, schedule)` / `validate_raw_schedule(jobs, operations, makespan)` | `jobs`, `DecodedScheduleResult` or raw op tuples | `bool` | Runs `jssp_core::feasibility::validate_schedule` in Rust. |
| `evaluate_makespan(jobs, schedule)` | `jobs`, `DecodedScheduleResult` | `int` | Recomputes exact `C_max` in Rust via `jssp_core::objective::evaluate_makespan`. |
| `get_frozen_master_seeds()` | None | `tuple[int, ...]` | Returns the 20 SHA-256 frozen confirmatory master seeds (`r = 0..19`). |
| `get_rng_metadata()` | None | `RngMetadataRecord` | Returns authoritative Rust RNG algorithm, pinned crate versions, rounds, schema, and roles. |
| `derive_rng_streams(...)` / `derive_rng_streams_with_seed(...)` | `replicate_index`, `instance_id`, pairing args, `preview_count` | `tuple[DerivedStreamPreview, ...]` | Derives all 9 D07-A stream SHA-256 seeds and preview `u64` draws in Rust. |
| `rng_sample_range_and_shuffle(seed_hex, upper, count, shuffle_len)` | 64-hex seed, bounds, lengths | `tuple[list[int], list[int]]` | Executes Rust ChaCha20 rejection range sampling and Fisher–Yates shuffle for verification/inspection. |

---

## 3. Data Serialization Schema

- **Instance Row Format:** `jobs[j]` is a sequence of `(machine_id, processing_time)` pairs using 0-based `int` machine indices and strictly positive integer durations (`1 <= p <= 2^64 - 1`).
- **Scheduled Operation Tuple (`RawScheduledOperationTuple`):**
  `(job, op_index, machine, start_time, completion_time, processing_time)` mapped in Python to frozen dataclass `ScheduledOperationRecord`.
- **Machine Sequences:** `tuple[tuple[tuple[int, int], ...], ...]` ordered by machine index `0..M-1` and processing interval order `(start_time, completion_time, job, op_index)`.

---

## 4. Error Translation Contract

Typed `jssp_core::error::JsspError` values returned from Rust are converted at the `crates/jssp-python` boundary without panicking:

- **`JsspError::TimeOverflow` & `JsspError::DimensionOverflow`** $\longrightarrow$ Python **`OverflowError`**
- **All remaining validation, OBR, precedence, overlap, coverage, metadata, and makespan errors** $\longrightarrow$ Python **`ValueError`** (preserving the structured `Display` diagnostic message)
- Out-of-range negative integers passed from Python fail PyO3 `usize`/`u64` conversion cleanly with `OverflowError` / `TypeError`.

---

## 5. Build & Development Workflow

After any edit under `crates/jssp-core` or `crates/jssp-python`, rebuild the in-place editable `.so` extension inside the pinned `uv` CPython 3.15.2 environment before running Python tests:

```bash
uv run --locked maturin develop --manifest-path crates/jssp-python/Cargo.toml
uv run --locked pytest
```
