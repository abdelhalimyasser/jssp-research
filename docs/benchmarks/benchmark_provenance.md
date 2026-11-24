# Benchmark Provenance & Committed Parser Fixtures (`M3-B`)

**Authoritative Fixture Manifest:** [`tests/fixtures/expected_schedules.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/tests/fixtures/expected_schedules.json)

---

## 1. Committed Local Fixture Checksums & Provenance

| Fixture ID | File / Embedded Location | Format | Dimensions | Provenance & License | Raw SHA-256 | Canonical Parsed SHA-256 |
|---|---|---|---|---|---|---|
| `tiny_jssp` | [`tiny_jssp.txt`](file:///home/abdelhalimyasser/Downloads/jssp-research/tests/fixtures/tiny_jssp.txt) | `standard_jssp_0based_v1` | `3 x 3` | Xie et al. (2023) Table 1 / [`obr_decoder.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/obr_decoder.md) §4 (retrieved `2026-11-23`; Academic research fixture) | `7cf72303e042fdfa980f7507cf5a1c4795291faf5cf5adc69940453d1ea192c7` | `29ca8e5c1acd6ff11cf73475c2a6bb62f24647f02b93211aca5f006734d09ee4` |
| `ft06` | [`medium_jssp.txt`](file:///home/abdelhalimyasser/Downloads/jssp-research/tests/fixtures/medium_jssp.txt) | `standard_jssp_0based_v1` | `6 x 6` | Fisher & Thompson (1963) / JSPLib `ft06` (retrieved `2026-11-23`; Public academic OR benchmark; proven optimum `55`) | `e8effb107e91728e101a3b257bd2d23c59bd398c75510df69650eea394370d48` | `1c2c4a9a1a5d13f5ec463e5f264e27c04499aadb6760038fb2aaa1ce98a74ad8` |
| `tiny_jssp_1based` | Embedded in [`expected_schedules.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/tests/fixtures/expected_schedules.json) | `standard_jssp_1based_v1` | `3 x 3` | 1-based machine encoding of `tiny_jssp` (`2026-11-23`) | `6f71319d4c60796965b7e423b2fc0e8743ca2a1602b01784ca9278b4a83d5d2a` | `29ca8e5c1acd6ff11cf73475c2a6bb62f24647f02b93211aca5f006734d09ee4` *(matches `tiny_jssp`)* |
| `ft06_taillard` | Embedded in [`expected_schedules.json`](file:///home/abdelhalimyasser/Downloads/jssp-research/tests/fixtures/expected_schedules.json) | `taillard_matrix_v1` | `6 x 6` | Taillard (1993) dual-matrix encoding of `ft06` (`2026-11-23`) | `81392b76856eee4383a192f4192bf23ab03601e8e5266f2e10573f578eabb378` | `1c2c4a9a1a5d13f5ec463e5f264e27c04499aadb6760038fb2aaa1ce98a74ad8` *(matches `ft06`)* |

---

## 2. Bounds Policy & Pending Snapshot Discipline

- `BoundProvenance` explicitly separates `PROVEN_OPTIMAL`, `BEST_KNOWN_OPEN`, and `PENDING_SNAPSHOT`.
- Proven optima are only recorded where verified (`ft06 = 55`, `ft10 = 930`, `ft20 = 1165` per Fisher & Thompson 1963 / Carlier & Pinson 1989 / JSPLib, recorded `2026-11-23`).
- All remaining external Track 1–5 instances carry `BoundStatus.PENDING_SNAPSHOT` with `None` bounds and `AcquisitionStatus.METADATA_CATALOGED_FILE_PENDING` (or `DEFERRED_UNSUPPORTED_MODEL`) until full raw files and a date-stamped JSPLib bounds snapshot are acquired in a dedicated data-acquisition step. Missing bounds or file checksums are never fabricated.

---

## 3. Track 3 — Strassl & Musliu (2022) Synthetic Sample Provenance (`M3.1 E04`)

- **Original Source Dataset Identity:** `strassl_musliu_2022_jssp_4225` (4,225 generated classical rectangular JSSP instances)
- **Dataset Version & Citation:** `v1` — S. Strassl and N. Musliu, *"Instance space analysis and algorithm selection for the job shop scheduling problem,"* *Computers & Operations Research*, vol. 141, 105661, 2022 (`DOI: 10.1016/j.cor.2021.105661`).
- **Source Verification Status & Archive Checksum:** `pending_raw_archive_acquisition` — `TRACK3_SOURCE_ARCHIVE_SHA256 = None`. Raw source files have **not** been acquired into this workspace yet; only pre-registered metadata and split rules are validated here.
- **Sampling Method & Pre-Registered Seed:**
  - Method: **`stratified_16_cell_equal_allocation_10_per_stratum`** (`16` feature strata $\times 10$ instances per stratum $= 160$ instances `sm001`–`sm160`).
  - Selection Seed Identifier: **`jssp-research-v1-track3-selection-seed (0x5433534D32303236)`**.
  - **Pre-Outcome Blinded Assignment:** `TRACK3_PRE_OUTCOME_BLINDED = True` — stratum layout, instance slots (`sm001`–`sm160`), and 40/120 split assignment were frozen completely prior to running any JSSP algorithm or inspecting solver outcomes.

### 3.1 Sixteen Feature Strata and Boundaries

Each stratum holds `10` consecutive catalog IDs: stratum `k = (idx - 1) // 10` crosses 4 size/aspect-ratio buckets (`S0`–`S3`) with 4 processing-time/load-skew regimes (`D0`–`D3`):

| Stratum ID | IDs (`10` each) | Size Regime | $J$ Bounds | $M$ Bounds | $J/M$ Ratio | Duration & Machine-Load Regime |
|---|---|---|---:|---:|---:|---|
| `SM_STRATUM_00` | `sm001`–`sm010` | `S0_small_square` | `[10, 15]` | `[10, 15]` | `[0.8, 1.25]` | `D0_uniform_narrow` ($p \in [1, 50]$, load $\mathrm{CV} < 0.10$) |
| `SM_STRATUM_01` | `sm011`–`sm020` | `S0_small_square` | `[10, 15]` | `[10, 15]` | `[0.8, 1.25]` | `D1_uniform_wide` ($p \in [1, 100]$, $0.10 \le \mathrm{CV} < 0.25$) |
| `SM_STRATUM_02` | `sm021`–`sm030` | `S0_small_square` | `[10, 15]` | `[10, 15]` | `[0.8, 1.25]` | `D2_bottleneck_skew` (bottleneck load $\ge 1.35\times$ mean) |
| `SM_STRATUM_03` | `sm031`–`sm040` | `S0_small_square` | `[10, 15]` | `[10, 15]` | `[0.8, 1.25]` | `D3_bimodal_cluster` ($p \in [1, 20] \cup [80, 100]$) |
| `SM_STRATUM_04` | `sm041`–`sm050` | `S1_tall_rect` | `[20, 40]` | `[10, 15]` | `[1.5, 3.0]` | `D0_uniform_narrow` |
| `SM_STRATUM_05` | `sm051`–`sm060` | `S1_tall_rect` | `[20, 40]` | `[10, 15]` | `[1.5, 3.0]` | `D1_uniform_wide` |
| `SM_STRATUM_06` | `sm061`–`sm070` | `S1_tall_rect` | `[20, 40]` | `[10, 15]` | `[1.5, 3.0]` | `D2_bottleneck_skew` |
| `SM_STRATUM_07` | `sm071`–`sm080` | `S1_tall_rect` | `[20, 40]` | `[10, 15]` | `[1.5, 3.0]` | `D3_bimodal_cluster` |
| `SM_STRATUM_08` | `sm081`–`sm090` | `S2_medium_square` | `[20, 30]` | `[20, 30]` | `[0.8, 1.25]` | `D0_uniform_narrow` |
| `SM_STRATUM_09` | `sm091`–`sm100` | `S2_medium_square` | `[20, 30]` | `[20, 30]` | `[0.8, 1.25]` | `D1_uniform_wide` |
| `SM_STRATUM_10` | `sm101`–`sm110` | `S2_medium_square` | `[20, 30]` | `[20, 30]` | `[0.8, 1.25]` | `D2_bottleneck_skew` |
| `SM_STRATUM_11` | `sm111`–`sm120` | `S2_medium_square` | `[20, 30]` | `[20, 30]` | `[0.8, 1.25]` | `D3_bimodal_cluster` |
| `SM_STRATUM_12` | `sm121`–`sm130` | `S3_large_tall` | `[40, 60]` | `[15, 20]` | `[2.0, 4.0]` | `D0_uniform_narrow` |
| `SM_STRATUM_13` | `sm131`–`sm140` | `S3_large_tall` | `[40, 60]` | `[15, 20]` | `[2.0, 4.0]` | `D1_uniform_wide` |
| `SM_STRATUM_14` | `sm141`–`sm150` | `S3_large_tall` | `[40, 60]` | `[15, 20]` | `[2.0, 4.0]` | `D2_bottleneck_skew` |
| `SM_STRATUM_15` | `sm151`–`sm160` | `S3_large_tall` | `[40, 60]` | `[15, 20]` | `[2.0, 4.0]` | `D3_bimodal_cluster` |

### 3.2 Exact 40 Development and 120 Held-Out Confirmatory IDs

- **Development Split (40 instances, `(idx - 1) % 4 == 0`):**
  `sm001, sm005, sm009, sm013, sm017, sm021, sm025, sm029, sm033, sm037, sm041, sm045, sm049, sm053, sm057, sm061, sm065, sm069, sm073, sm077, sm081, sm085, sm089, sm093, sm097, sm101, sm105, sm109, sm113, sm117, sm121, sm125, sm129, sm133, sm137, sm141, sm145, sm149, sm153, sm157`
- **Held-Out Confirmatory Split (120 instances, `TRACK3_ALL_IDS \ TRACK3_DEVELOPMENT_IDS`):**
  `sm002, sm003, sm004, sm006, sm007, sm008, sm010, sm011, sm012, sm014, sm015, sm016, sm018, sm019, sm020, sm022, sm023, sm024, sm026, sm027, sm028, sm030, sm031, sm032, sm034, sm035, sm036, sm038, sm039, sm040, sm042, sm043, sm044, sm046, sm047, sm048, sm050, sm051, sm052, sm054, sm055, sm056, sm058, sm059, sm060, sm062, sm063, sm064, sm066, sm067, sm068, sm070, sm071, sm072, sm074, sm075, sm076, sm078, sm079, sm080, sm082, sm083, sm084, sm086, sm087, sm088, sm090, sm091, sm092, sm094, sm095, sm096, sm098, sm099, sm100, sm102, sm103, sm104, sm106, sm107, sm108, sm110, sm111, sm112, sm114, sm115, sm116, sm118, sm119, sm120, sm122, sm123, sm124, sm126, sm127, sm128, sm130, sm131, sm132, sm134, sm135, sm136, sm138, sm139, sm140, sm142, sm143, sm144, sm146, sm147, sm148, sm150, sm151, sm152, sm154, sm155, sm156, sm158, sm159, sm160`
