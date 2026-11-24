# Deterministic Named PRNG Architecture (`M3-A`)

**Implementation:** [`crates/jssp-core/src/rng/streams.rs`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/rng/streams.rs), [`src/jssp_research/experiments/seeds.py`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/experiments/seeds.py)

---

## 1. Portable PRNG Engine Choice

- **Generator Algorithm:** `ChaCha20Rng` (20-round ChaCha counter-based stream cipher PRNG). *(Supersedes the earlier `ChaCha8Rng` / `SplitMix64` informal mention in `fairness_protocol.md` in accordance with the M3 `ChaCha20Rng` specification.)*
- **Pinned Rust Crate Versions:**
  - `rand_chacha = "=0.10.0"` (`rand_core = "0.10.1"`)
  - `sha2 = "=0.11.1"` (`digest = "0.11.3"`)
- **Seed Representation:** 256-bit byte array (`[u8; 32]`, formatted in manifests as lowercase 64-character hexadecimal).
- **Prohibited Sources:** No thread-local global RNG, `thread_rng`, `StdRng`, OS entropy during experiment execution, or single shared mutable PRNG across algorithmic roles.

---

## 2. Unambiguous Domain-Separated Seed Derivation

For each run context `(schema_version, master_seed, replicate_index, instance_id, role, coupling_key)`, a 256-bit seed is computed via **SHA-256** over an unambiguous length-prefixed little-endian binary frame:

```text
b"JSSP_RNG_V1\0"
  || len_le32(schema_version) || utf8(schema_version)   // "jssp-rng-derivation-v1"
  || le64(master_seed)
  || le32(replicate_index)
  || len_le32(instance_id)    || utf8(instance_id)
  || len_le32(role.as_str())  || utf8(role.as_str())
  || len_le32(coupling_key)   || utf8(coupling_key)
```

Because each variable-length field is preceded by an explicit 32-bit little-endian byte length, ambiguity from delimiter collisions or raw string concatenation is impossible. Note carefully: **SHA-256 domain separation guarantees computational separation of seed material and eliminates construction-order/consumption-order interference between streams**; this is computational/algorithmic independence rather than an unconditional information-theoretic claim.

---

## 3. Project-Owned Versioned Random Primitives

Even with a pinned PRNG (`ChaCha20Rng`), third-party `rand::distributions` or `SliceRandom` implementations can change internal algorithms across library releases. Therefore `jssp-core` owns and pins its sampling and shuffling algorithms:

1. **Raw Word Generation (`next_u64`):** Direct 64-bit little-endian output from `ChaCha20Rng::try_next_u64()`, tracking per-stream `draws_u64`.
2. **Unbiased Integer Range Sampling (`gen_range_u64`, `gen_range_usize`, `choose_index`):** Modulo rejection sampling over `[0, upper)` using exact divisibility threshold `threshold = upper.wrapping_neg() % upper` (`x >= threshold -> Ok(x % upper)`), rejecting `upper == 0` with `JsspError::InvalidRandomBound`. Zero bias for arbitrary non-power-of-two ranges.
3. **Fisher–Yates Shuffle (`shuffle`):** Project-owned top-down Knuth/Fisher–Yates in-place shuffle for `i` down from `len - 1` to `1`, drawing index `j in [0, i + 1)` via `gen_range_usize(i + 1)`.
4. **Sampling Without Replacement (`sample_indices_without_replacement`):** Project-owned partial Fisher–Yates over `0..population_len` for `k` swaps, rejecting `k > population_len` with `JsspError::InvalidSampleSize`.

**Limitation / Guarantee Scope:** Reproducibility holds across repeated runs and platforms under the pinned crate versions (`rand_chacha = "=0.10.0"`, `sha2 = "=0.11.1"`) and `jssp-core` implementation. Any future bump to sampling primitives or ChaCha rounds must increment `SEED_DERIVATION_SCHEMA_V1`.

---

## 4. Nine Named Stream Roles (`D07-A`)

`RngStreamRole` preserves all nine approved `D07-A` roles verbatim:
`"init"`, `"qmc_scramble"`, `"selection"`, `"crossover"`, `"mutation"`, `"local_search_sample"`, `"perturbation"`, `"sa_accept"`, `"mh_filter"`.

Each role derives an isolated `ChaCha20Rng` instance independently. Creating streams in arbitrary order or drawing any number of samples from one stream never advances or alters another stream.

---

## 5. Shared Stream Seeds vs. Event-Aligned Common Random Numbers (`E01`)

Sharing a seed across two algorithm configurations must be distinguished carefully from **event-aligned Common Random Numbers (CRN)**:

1. **Event-Aligned Pairing (`paired_init`, `PairingPolicy::PairedInitialAndScramble`):** Holds strictly for fixed-draw-schedule pre-search stages where both configurations execute identical logical steps (`"init"` and `"qmc_scramble"` with matching `(instance_id, replicate_index, pairing_group)`). For example, both `B01` and `B02` consume the exact same `Init` draws to construct identical initial OBR schedules.
2. **Shared Named-Stream Seeds (`shared_stream_seeds`, `PairingPolicy::SharedStreamSeeds`; deprecated alias `"crn"`):** Assigns identical initial 256-bit ChaCha20 seeds for all 9 named streams within a `pairing_group`. However, if Configuration A and Configuration B execute different branching logic inside downstream search (e.g., Steepest Descent vs. Simulated Annealing vs. First Improvement), their `LocalSearchSample` / `Mutation` / `SaAccept` cursors advance by different counts and align with completely different logical events. Consequently, `shared_stream_seeds` provides **shared stream seeds**, NOT trajectory-wide event-aligned CRN.
3. **Configuration-Scoped (`config_scoped`, `PairingPolicy::ConfigurationScoped`):** Separates all 9 streams per `configuration_id` when experimental independence across configurations is intended (e.g., stochastic Phase A initializers).
