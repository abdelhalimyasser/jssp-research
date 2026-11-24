# Frozen Seed Schedule & Paired Experimental Coupling Protocol (`M3-A`)

**Status:** Active Specification (`MASTER_SEED_SCHEDULE_ID_V1 = "jssp-research-v1-sha256-20"`)

---

## 1. Frozen 20-Replicate Confirmatory Master Seeds

Matching [`docs/experiments/fairness_protocol.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/fairness_protocol.md) §2, each confirmatory run replicate $r \in \{0, \dots, 19\}$ resolves its deterministic 64-bit unsigned `master_seed` from:

$$\mathrm{master\_seed}(r) = \mathrm{u64\_le}(\mathrm{SHA256}(\text{"jssp-research-v1-seed-"} \| r)[0..8])$$

| Replicate $r$ | Hex (`u64`) | Replicate $r$ | Hex (`u64`) |
|---:|---|---:|---|
| `0` | `0x9BAAFAE1C0E301EF` | `10` | `0xD9C09BF880B30E25` |
| `1` | `0x5B321F3CC67EB287` | `11` | `0xF0816B36E7DF6FED` |
| `2` | `0x7707941F9049A683` | `12` | `0x0B23F874CFCBE8AA` |
| `3` | `0xB0C4495E1FD0FEBC` | `13` | `0x6A35950DD8C59F95` |
| `4` | `0x364BD139127D5405` | `14` | `0x23562A029FD284B1` |
| `5` | `0x3E474AE565AE5533` | `15` | `0x1F0AD8FFD61A0535` |
| `6` | `0xF973472382E84120` | `16` | `0xB72A40A1C7CE2E33` |
| `7` | `0x17B913836B127E01` | `17` | `0x516C009450C6D466` |
| `8` | `0x8B8A319EC341F85F` | `18` | `0xBF4FD20A8BE1DC48` |
| `9` | `0x15E81F29E54302E5` | `19` | `0xC45021E1951EFC26` |

*(The arithmetic array `[101, 202, ..., 2020]` mentioned as an alternative in `fairness_protocol.md` is archived as `ARITHMETIC_FALLBACK_SEEDS_V1` only and is not the active confirmatory schedule.)*

---

## 2. Paired Experimental Comparison Semantics (`E01`)

Simply hashing `configuration_id` into every stream destroys paired initial blocking when comparing configurations against the same starting schedule or QMC scramble. Conversely, merely giving two different search algorithms identical downstream stream seeds does **not** constitute true **event-aligned Common Random Numbers (CRN)** when their branching trajectories consume different numbers of variates for different logical events.

`PairingPolicy` (`PairingMode` in Python) distinguishes these regimes explicitly:

1. **`paired_init` (`PairingPolicy::PairedInitialAndScramble` — Default for Phase B & Phase C):**
   - `init` and `qmc_scramble` receive `coupling_key = "paired:{pairing_group}"`. Two configurations (`B01` vs. `B02`, or `C01` vs. `C06`) sharing the same `(replicate_index, instance_id, pairing_group)` execute the exact same initial-generation / scramble event sequence and thus realize **true event-aligned pairing** on starting states.
   - `selection`, `crossover`, `mutation`, `local_search_sample`, `perturbation`, `sa_accept`, and `mh_filter` receive `coupling_key = "config:{configuration_id}"`, isolating stochastic search decisions per configuration.
2. **`shared_stream_seeds` (`PairingPolicy::SharedStreamSeeds`; backward-compatible CLI/Python alias `"crn"`):**
   - All 9 streams share `coupling_key = "paired:{pairing_group}"`. This guarantees **identical initial named-stream seeds** across configurations, **not** full trajectory-level event-aligned CRN if algorithms advance stream cursors at different rates. Event-aligned CRN holds only when two compared algorithms consume identical variate positions for corresponding logical events.
3. **`config_scoped` (`PairingPolicy::ConfigurationScoped`):**
   - All 9 streams use `coupling_key = "config:{configuration_id}"`, yielding strictly separated stream seed derivations between configurations.
