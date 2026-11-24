# Benchmark Catalog (`M3-B`)

**Implementation:** [`src/jssp_research/benchmarks/catalog.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/benchmarks/catalog.py), [`src/jssp_research/benchmarks/splits.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/benchmarks/splits.py)

---

## 1. Cataloged Tracks vs. Locally Acquired Files

Important distinction (`M3.1 E04`): cataloging metadata for all **536 planned instances** validates metadata schemas, unique IDs, parser references, deferral boundaries, and split disjointness (`metadata_entries_validated = 536`), whereas **actual benchmark-file parsing and SHA-256 verification** applies **only** to files physically present on disk (`local_files_parsed_and_verified = 1` cataloged entry `ft06` in `medium_jssp.txt`, plus standalone fixture `tiny_jssp.txt` and two embedded format fixtures in `expected_schedules.json`). Unacquired external files (`491` pending entries across Tracks 1–3) carry `AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING` with `source_verification_status = "pending_raw_archive_acquisition"`.

| Track | Families | Total Cataloged | Split Breakdown | Local Files Available (M3.1) | Status |
|---|---|---:|---|---:|---|
| **Track 1** | `FT` (3), `LA` (40), `ABZ` (5), `ORB` (10), `SWV` (20), `YN` (4), `TA` (80), `DMU` (80) | **242** | **48** Dev / **194** Confirmatory | **1** (`ft06` in `medium_jssp.txt`) *(+ standalone `tiny_jssp.txt`)* | Metadata validated; `241` raw files pending |
| **Track 2a** | `LARGE_TA` (`lta01`–`lta80`, $< 1000\times 1000$) | **80** | **80** `scalability_held_out` | 0 | Metadata validated; raw files pending |
| **Track 2b** | `STRESS_TAI` (`lta81`–`lta90`, $1000\times 1000$) | **10** | **10** `stress_held_out` | 0 | Metadata validated; raw files pending |
| **Track 3** | `STRASSL_MUSLIU` (`sm001`–`sm160` of 4,225) | **160** | **40** Dev / **120** Confirmatory | 0 | Metadata validated; archive SHA-256 `None` (`pending_raw_archive_acquisition`) |
| **Track 4** | `DCT_VARIABLE_ROUTE` (`dct_ko01`–`dct_ko24`) | **24** | **24** `deferred` | 0 | **DEFERRED (`D07-C`)** |
| **Track 5** | `BEL_INDUSTRIAL` (`bel01`–`bel20`) | **20** | **20** `deferred` | 0 | **DEFERRED (`D07-C`)** |
| **Total** | **13 Cataloged Families** | **536** | — | **1 cataloged + 1 standalone fixture** | **Metadata validated (file checks pending acquisition)** |

---

## 2. Track 1 Family-Stratified 48 / 194 Split

Every classical Track 1 family contributes to the **48-instance Development Split** (`TRACK1_DEVELOPMENT_IDS`):
- `FT`: `1 / 3` (`ft06`)
- `LA`: `8 / 40` (`la01, la06, la11, la16, la21, la26, la31, la36`)
- `ABZ`: `1 / 5` (`abz5`)
- `ORB`: `2 / 10` (`orb01, orb06`)
- `SWV`: `4 / 20` (`swv01, swv06, swv11, swv16`)
- `YN`: `1 / 4` (`yn1`)
- `TA`: `16 / 80` (2 per 10-instance stratum)
- `DMU`: `15 / 80` (across all 8 size strata)

---

## 3. Supported Parsers (`PARSER_REGISTRY`)

1. **`standard_jssp_0based_v1` (v1.0.0):** Rectangular JSPLib / OR-Library `<J> <M>` header followed by $J$ rows of $2M$ `(machine_0based, duration)` pairs.
2. **`standard_jssp_1based_v1` (v1.0.0):** Interleaved rectangular format with 1-based machine indices ($1 \dots M$).
3. **`taillard_matrix_v1` (v1.0.0):** Taillard (1993) dual-matrix format (`Times` matrix + 1-based `Machines` matrix).

All parsers normalize into 0-based `ParsedJsspInstance` and pass mandatory Rust domain validation (`jssp_research.native.validate_instance`).
