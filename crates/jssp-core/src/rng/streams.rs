//! Deterministic named ChaCha20 PRNG substreams and domain-separated seed derivation.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{SeedableRng, TryRng};
use sha2::{Digest, Sha256};

use crate::error::JsspError;

/// Portable underlying PRNG algorithm name (`ChaCha` with 20 rounds).
pub const RNG_ALGORITHM_ID: &str = "ChaCha20Rng";

/// Exact crate version backing the portable PRNG engine.
pub const RNG_CRATE_VERSION: &str = "rand_chacha-0.10.0";

/// Exact SHA-256 crate version backing domain-separated seed derivation.
pub const HASH_CRATE_VERSION: &str = "sha2-0.11.1";

/// Number of ChaCha core rounds used per block generation.
pub const CHACHA_ROUNDS: u32 = 20;

/// Canonical binary seed-derivation schema identifier.
pub const SEED_DERIVATION_SCHEMA_V1: &str = "jssp-rng-derivation-v1";

/// Identifier for the primary 20-replicate frozen SHA-256 master seed schedule.
pub const MASTER_SEED_SCHEDULE_ID_V1: &str = "jssp-research-v1-sha256-20";

/// Total number of replicates in the frozen confirmatory master-seed schedule.
pub const NUM_CONFIRMATORY_SEEDS: u32 = 20;

/// Frozen 20 confirmatory master seeds derived via little-endian 64-bit truncation of
/// `SHA256("jssp-research-v1-seed-{r}")` for `r in 0..20`.
pub const FROZEN_MASTER_SEEDS_V1: [u64; 20] = [
    0x9BAA_FAE1_C0E3_01EF, // r=0
    0x5B32_1F3C_C67E_B287, // r=1
    0x7707_941F_9049_A683, // r=2
    0xB0C4_495E_1FD0_FEBC, // r=3
    0x364B_D139_127D_5405, // r=4
    0x3E47_4AE5_65AE_5533, // r=5
    0xF973_4723_82E8_4120, // r=6
    0x17B9_1383_6B12_7E01, // r=7
    0x8B8A_319E_C341_F85F, // r=8
    0x15E8_1F29_E543_02E5, // r=9
    0xD9C0_9BF8_80B3_0E25, // r=10
    0xF081_6B36_E7DF_6FED, // r=11
    0x0B23_F874_CFCB_E8AA, // r=12
    0x6A35_950D_D8C5_9F95, // r=13
    0x2356_2A02_9FD2_84B1, // r=14
    0x1F0A_D8FF_D61A_0535, // r=15
    0xB72A_40A1_C7CE_2E33, // r=16
    0x516C_0094_50C6_D466, // r=17
    0xBF4F_D20A_8BE1_DC48, // r=18
    0xC450_21E1_951E_FC26, // r=19
];

/// Secondary documented deterministic fallback schedule `[101, 202, ..., 2020]`
/// referenced in `docs/experiments/fairness_protocol.md` (not active default).
pub const ARITHMETIC_FALLBACK_SEEDS_V1: [u64; 20] = [
    101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717,
    1818, 1919, 2020,
];

/// Computes the frozen SHA-256-derived master seed for replicate `r` from ASCII label
/// `format!("jssp-research-v1-seed-{replicate_index}")`.
///
/// # Errors
/// Returns [`JsspError::InvalidReplicateIndex`] if `replicate_index >= NUM_CONFIRMATORY_SEEDS`.
pub fn compute_frozen_master_seed(replicate_index: u32) -> Result<u64, JsspError> {
    if replicate_index >= NUM_CONFIRMATORY_SEEDS {
        return Err(JsspError::InvalidReplicateIndex {
            replicate_index,
            max_replicates: NUM_CONFIRMATORY_SEEDS,
        });
    }
    let label = format!("jssp-research-v1-seed-{replicate_index}");
    let digest = Sha256::digest(label.as_bytes());
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    Ok(u64::from_le_bytes(prefix))
}

/// Returns the frozen confirmatory master seed for `replicate_index in 0..20`.
///
/// # Errors
/// Returns [`JsspError::InvalidReplicateIndex`] if `replicate_index >= 20`.
pub fn master_seed_for_replicate(replicate_index: u32) -> Result<u64, JsspError> {
    FROZEN_MASTER_SEEDS_V1
        .get(replicate_index as usize)
        .copied()
        .ok_or(JsspError::InvalidReplicateIndex {
            replicate_index,
            max_replicates: NUM_CONFIRMATORY_SEEDS,
        })
}

/// The nine approved isolated RNG substream roles specified by Decision `D07-A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RngStreamRole {
    /// Random OBR / Monte Carlo+MWR population initialization.
    Init,
    /// Randomized Halton digit permutation and Sobol Matousek scrambling.
    QmcScramble,
    /// Evolutionary parent tournament selection.
    Selection,
    /// Order-preserving crossover operator.
    Crossover,
    /// Chromosome mutation loci and operator selection.
    Mutation,
    /// Random Improvement descent, SA move sampling, and Tabu dynamic tenure.
    LocalSearchSample,
    /// ILS and Tabu diversification 3-cycle OBR perturbations.
    Perturbation,
    /// Simulated Annealing Metropolis acceptance thresholds.
    SaAccept,
    /// Hybrid ILS+MH pre-descent perturbation acceptance filter.
    MhFilter,
}

impl RngStreamRole {
    /// Canonical list of all nine approved `D07-A` stream roles in fixed definition order.
    pub const ALL: [Self; 9] = [
        Self::Init,
        Self::QmcScramble,
        Self::Selection,
        Self::Crossover,
        Self::Mutation,
        Self::LocalSearchSample,
        Self::Perturbation,
        Self::SaAccept,
        Self::MhFilter,
    ];

    /// Canonical lowercase ASCII identifier stored in seed derivation frames and manifests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::QmcScramble => "qmc_scramble",
            Self::Selection => "selection",
            Self::Crossover => "crossover",
            Self::Mutation => "mutation",
            Self::LocalSearchSample => "local_search_sample",
            Self::Perturbation => "perturbation",
            Self::SaAccept => "sa_accept",
            Self::MhFilter => "mh_filter",
        }
    }

    /// Parses a canonical D07 stream role identifier.
    ///
    /// # Errors
    /// Returns [`JsspError::UnknownRngStreamRole`] for unrecognised identifiers.
    pub fn from_name(name: &str) -> Result<Self, JsspError> {
        Self::ALL
            .iter()
            .copied()
            .find(|role| role.as_str() == name)
            .ok_or_else(|| JsspError::UnknownRngStreamRole {
                name: name.to_owned(),
            })
    }

    /// Returns `true` for streams that share a paired baseline across comparable
    /// configurations by default (`init` and `qmc_scramble`).
    #[must_use]
    pub const fn is_paired_by_default(self) -> bool {
        matches!(self, Self::Init | Self::QmcScramble)
    }
}

/// Seed-coupling policy governing whether a stream shares initial seed material
/// across configurations via a pairing group or is isolated per configuration ID.
///
/// Note: `SharedStreamSeeds` assigns identical initial 256-bit ChaCha20 seeds
/// across configurations in the same `pairing_group`. This guarantees **shared
/// named-stream seeds**, not full **event-aligned Common Random Numbers (CRN)**,
/// because distinct search algorithms consume different numbers of draws and reach
/// different logical branching events downstream. True event alignment holds only
/// for fixed-draw pre-search stages (`init` and `qmc_scramble` under
/// [`PairingPolicy::PairedInitialAndScramble`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingPolicy {
    /// Shared named-stream seeds across all 9 roles via `pairing_group`
    /// (does not imply trajectory-level event alignment in search loops).
    SharedStreamSeeds { pairing_group: String },
    /// Hybrid paired coupling (default for Phase B & Phase C): `init` and `qmc_scramble`
    /// share `pairing_group` (exact event-aligned pairing of initial schedule / scramble),
    /// while the 7 search/operator streams scope independently to `configuration_id`.
    PairedInitialAndScramble {
        pairing_group: String,
        configuration_id: String,
    },
    /// Independent configuration-scoped streams: all 9 streams scope to `configuration_id`.
    ConfigurationScoped { configuration_id: String },
}

impl PairingPolicy {
    /// Resolves the canonical coupling tag included in SHA-256 stream seed derivation.
    #[must_use]
    pub fn coupling_key_for_role(&self, role: RngStreamRole) -> String {
        match self {
            Self::SharedStreamSeeds { pairing_group } => format!("paired:{pairing_group}"),
            Self::PairedInitialAndScramble {
                pairing_group,
                configuration_id,
            } => {
                if role.is_paired_by_default() {
                    format!("paired:{pairing_group}")
                } else {
                    format!("config:{configuration_id}")
                }
            }
            Self::ConfigurationScoped { configuration_id } => {
                format!("config:{configuration_id}")
            }
        }
    }

    /// Stable human-readable policy identifier recorded in manifests and FFI payloads.
    #[must_use]
    pub const fn policy_name(&self) -> &'static str {
        match self {
            Self::SharedStreamSeeds { .. } => "shared_stream_seeds",
            Self::PairedInitialAndScramble { .. } => "paired_init",
            Self::ConfigurationScoped { .. } => "config_scoped",
        }
    }
}

/// Complete domain-separated derivation context for one experimental run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedDerivationContext {
    /// Derivation frame schema version (`jssp-rng-derivation-v1`).
    pub schema_version: String,
    /// Master 64-bit seed for the run replicate.
    pub master_seed: u64,
    /// Replicate index associated with the run.
    pub replicate_index: u32,
    /// Canonical benchmark instance identifier (e.g. `ft06`).
    pub instance_id: String,
    /// Pairing policy controlling shared vs configuration-specific coupling keys.
    pub pairing_policy: PairingPolicy,
}

impl SeedDerivationContext {
    /// Constructs a context using the frozen master seed for `replicate_index in 0..20`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidReplicateIndex`] if `replicate_index >= 20`, or
    /// [`JsspError::InvalidRngConfiguration`] if `instance_id` is empty.
    pub fn from_frozen_replicate(
        replicate_index: u32,
        instance_id: impl Into<String>,
        pairing_policy: PairingPolicy,
    ) -> Result<Self, JsspError> {
        let master_seed = master_seed_for_replicate(replicate_index)?;
        Self::with_explicit_master_seed(master_seed, replicate_index, instance_id, pairing_policy)
    }

    /// Constructs a context with an explicit 64-bit `master_seed` (e.g. pilot/synthetic tests).
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidRngConfiguration`] if `instance_id` is empty.
    pub fn with_explicit_master_seed(
        master_seed: u64,
        replicate_index: u32,
        instance_id: impl Into<String>,
        pairing_policy: PairingPolicy,
    ) -> Result<Self, JsspError> {
        let instance_id = instance_id.into();
        if instance_id.is_empty() {
            return Err(JsspError::InvalidRngConfiguration {
                detail: "instance_id must not be empty".to_owned(),
            });
        }
        Ok(Self {
            schema_version: SEED_DERIVATION_SCHEMA_V1.to_owned(),
            master_seed,
            replicate_index,
            instance_id,
            pairing_policy,
        })
    }

    /// Derives the unambiguous 256-bit ChaCha20 seed and effective coupling key for `role`.
    #[must_use]
    pub fn derive_seed(&self, role: RngStreamRole) -> ([u8; 32], String) {
        let coupling_key = self.pairing_policy.coupling_key_for_role(role);
        let seed_bytes = derive_stream_seed_bytes(
            &self.schema_version,
            self.master_seed,
            self.replicate_index,
            &self.instance_id,
            role,
            &coupling_key,
        );
        (seed_bytes, coupling_key)
    }

    /// Derives an independent [`NamedRngStream`] for `role`.
    #[must_use]
    pub fn stream(&self, role: RngStreamRole) -> NamedRngStream {
        let (seed_bytes, coupling_key) = self.derive_seed(role);
        NamedRngStream::from_raw_seed(role, seed_bytes, coupling_key)
    }

    /// Materializes all nine independent `D07-A` streams in canonical order.
    #[must_use]
    pub fn all_streams(&self) -> RunRngStreams {
        RunRngStreams::from_context(self)
    }
}

fn update_len_prefixed_str(hasher: &mut Sha256, value: &str) {
    let bytes = value.as_bytes();
    let len_u32 = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    hasher.update(len_u32.to_le_bytes());
    hasher.update(bytes);
}

/// Derives a 256-bit domain-separated seed via SHA-256 over a length-prefixed
/// canonical little-endian binary frame.
#[must_use]
pub fn derive_stream_seed_bytes(
    schema_version: &str,
    master_seed: u64,
    replicate_index: u32,
    instance_id: &str,
    role: RngStreamRole,
    coupling_key: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"JSSP_RNG_V1\0");
    update_len_prefixed_str(&mut hasher, schema_version);
    hasher.update(master_seed.to_le_bytes());
    hasher.update(replicate_index.to_le_bytes());
    update_len_prefixed_str(&mut hasher, instance_id);
    update_len_prefixed_str(&mut hasher, role.as_str());
    update_len_prefixed_str(&mut hasher, coupling_key);

    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    seed
}

/// Formats a 256-bit seed slice as lowercase 64-character hexadecimal.
#[must_use]
pub fn seed_to_hex(seed: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in seed {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0F) as usize] as char);
    }
    out
}

/// Parses a 64-character hexadecimal string into a 256-bit seed array.
///
/// # Errors
/// Returns [`JsspError::InvalidRngConfiguration`] if length or hex characters are invalid.
pub fn hex_to_seed(hex_str: &str) -> Result<[u8; 32], JsspError> {
    let bytes = hex_str.as_bytes();
    if bytes.len() != 64 {
        return Err(JsspError::InvalidRngConfiguration {
            detail: format!("expected 64-char hex seed, got {} chars", bytes.len()),
        });
    }
    let mut out = [0u8; 32];
    for (idx, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let hi = decode_hex_nibble(pair[0])?;
        let lo = decode_hex_nibble(pair[1])?;
        out[idx] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_hex_nibble(b: u8) -> Result<u8, JsspError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(JsspError::InvalidRngConfiguration {
            detail: format!("invalid hexadecimal character byte {b:#04x}"),
        }),
    }
}

/// Isolated portable `ChaCha20Rng` substream with versioned unbiased sampling primitives.
#[derive(Debug, Clone)]
pub struct NamedRngStream {
    role: RngStreamRole,
    seed_bytes: [u8; 32],
    coupling_key: String,
    rng: ChaCha20Rng,
    draws_u64: u64,
}

impl NamedRngStream {
    /// Constructs a stream directly from its role, derived 256-bit seed, and coupling key.
    #[must_use]
    pub fn from_raw_seed(
        role: RngStreamRole,
        seed_bytes: [u8; 32],
        coupling_key: impl Into<String>,
    ) -> Self {
        Self {
            role,
            seed_bytes,
            coupling_key: coupling_key.into(),
            rng: ChaCha20Rng::from_seed(seed_bytes),
            draws_u64: 0,
        }
    }

    /// Stream role identifier.
    #[must_use]
    pub const fn role(&self) -> RngStreamRole {
        self.role
    }

    /// Raw 256-bit ChaCha20 seed bytes.
    #[must_use]
    pub const fn seed_bytes(&self) -> &[u8; 32] {
        &self.seed_bytes
    }

    /// Lowercase hexadecimal encoding of the 256-bit stream seed.
    #[must_use]
    pub fn seed_hex(&self) -> String {
        seed_to_hex(&self.seed_bytes)
    }

    /// Effective domain-separation coupling tag used to derive this stream.
    #[must_use]
    pub fn coupling_key(&self) -> &str {
        &self.coupling_key
    }

    /// Count of raw 64-bit ChaCha20 outputs consumed from this stream so far.
    #[must_use]
    pub const fn draws_u64(&self) -> u64 {
        self.draws_u64
    }

    /// Generates the next portable 64-bit unsigned integer from `ChaCha20Rng`.
    pub fn next_u64(&mut self) -> u64 {
        self.draws_u64 = self.draws_u64.saturating_add(1);
        match self.rng.try_next_u64() {
            Ok(val) => val,
            Err(infallible) => match infallible {},
        }
    }

    /// Project-owned unbiased uniform integer sampling in half-open `[0, upper)`
    /// via Lemire-style / modulo rejection sampling (`u64::MAX + 1` divisibility threshold).
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidRandomBound`] when `upper == 0`.
    pub fn gen_range_u64(&mut self, upper: u64) -> Result<u64, JsspError> {
        if upper == 0 {
            return Err(JsspError::InvalidRandomBound { upper: 0 });
        }
        if upper == 1 {
            // Consume zero randomness or one word? No, standard uniform over singleton {0}
            // needs no rejection, and avoids modulo division issues cleanly.
            return Ok(0);
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return Ok(x % upper);
            }
        }
    }

    /// Project-owned unbiased uniform integer sampling in half-open `[0, upper)` as `usize`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidRandomBound`] if `upper == 0` or overflows `u64`.
    pub fn gen_range_usize(&mut self, upper: usize) -> Result<usize, JsspError> {
        let upper_u64 =
            u64::try_from(upper).map_err(|_| JsspError::InvalidRandomBound { upper: u64::MAX })?;
        let sampled = self.gen_range_u64(upper_u64)?;
        Ok(usize::try_from(sampled).unwrap_or(0))
    }

    /// Selects a uniform random index from finite non-empty set `0..len`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidRandomBound`] when `len == 0`.
    pub fn choose_index(&mut self, len: usize) -> Result<usize, JsspError> {
        self.gen_range_usize(len)
    }

    /// Project-owned deterministic top-down Fisher–Yates in-place shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range_usize(i + 1).unwrap_or(0);
            slice.swap(i, j);
        }
    }

    /// Project-owned reproducible sampling of `k` distinct indices from `0..population_len`
    /// without replacement via partial Fisher–Yates shuffle.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidSampleSize`] if `k > population_len`.
    pub fn sample_indices_without_replacement(
        &mut self,
        population_len: usize,
        k: usize,
    ) -> Result<Vec<usize>, JsspError> {
        if k > population_len {
            return Err(JsspError::InvalidSampleSize {
                population_len,
                requested: k,
            });
        }
        let mut pool: Vec<usize> = (0..population_len).collect();
        for i in 0..k {
            let offset = self.gen_range_usize(population_len - i)?;
            pool.swap(i, i + offset);
        }
        pool.truncate(k);
        Ok(pool)
    }
}

/// Complete bundle of all nine isolated `D07-A` streams for a single run context.
#[derive(Debug, Clone)]
pub struct RunRngStreams {
    /// Stream for random / MC-MWR initialization.
    pub init: NamedRngStream,
    /// Stream for randomized Halton and Matousek Sobol scrambling.
    pub qmc_scramble: NamedRngStream,
    /// Stream for GA tournament selection.
    pub selection: NamedRngStream,
    /// Stream for GA crossover.
    pub crossover: NamedRngStream,
    /// Stream for GA mutation.
    pub mutation: NamedRngStream,
    /// Stream for Random Improvement, SA sampling, and Tabu dynamic tenure.
    pub local_search_sample: NamedRngStream,
    /// Stream for ILS and Tabu diversification perturbations.
    pub perturbation: NamedRngStream,
    /// Stream for Simulated Annealing Metropolis acceptance draws.
    pub sa_accept: NamedRngStream,
    /// Stream for Hybrid ILS+MH pre-descent perturbation acceptance draws.
    pub mh_filter: NamedRngStream,
}

impl RunRngStreams {
    /// Deterministically derives all nine streams from `context`.
    #[must_use]
    pub fn from_context(context: &SeedDerivationContext) -> Self {
        Self {
            init: context.stream(RngStreamRole::Init),
            qmc_scramble: context.stream(RngStreamRole::QmcScramble),
            selection: context.stream(RngStreamRole::Selection),
            crossover: context.stream(RngStreamRole::Crossover),
            mutation: context.stream(RngStreamRole::Mutation),
            local_search_sample: context.stream(RngStreamRole::LocalSearchSample),
            perturbation: context.stream(RngStreamRole::Perturbation),
            sa_accept: context.stream(RngStreamRole::SaAccept),
            mh_filter: context.stream(RngStreamRole::MhFilter),
        }
    }

    /// Returns a shared reference to the requested named stream.
    #[must_use]
    pub const fn get(&self, role: RngStreamRole) -> &NamedRngStream {
        match role {
            RngStreamRole::Init => &self.init,
            RngStreamRole::QmcScramble => &self.qmc_scramble,
            RngStreamRole::Selection => &self.selection,
            RngStreamRole::Crossover => &self.crossover,
            RngStreamRole::Mutation => &self.mutation,
            RngStreamRole::LocalSearchSample => &self.local_search_sample,
            RngStreamRole::Perturbation => &self.perturbation,
            RngStreamRole::SaAccept => &self.sa_accept,
            RngStreamRole::MhFilter => &self.mh_filter,
        }
    }

    /// Returns a mutable reference to the requested named stream.
    pub fn get_mut(&mut self, role: RngStreamRole) -> &mut NamedRngStream {
        match role {
            RngStreamRole::Init => &mut self.init,
            RngStreamRole::QmcScramble => &mut self.qmc_scramble,
            RngStreamRole::Selection => &mut self.selection,
            RngStreamRole::Crossover => &mut self.crossover,
            RngStreamRole::Mutation => &mut self.mutation,
            RngStreamRole::LocalSearchSample => &mut self.local_search_sample,
            RngStreamRole::Perturbation => &mut self.perturbation,
            RngStreamRole::SaAccept => &mut self.sa_accept,
            RngStreamRole::MhFilter => &mut self.mh_filter,
        }
    }

    /// Returns metadata snapshots `(role, coupling_key, seed_hex)` across all 9 streams.
    #[must_use]
    pub fn metadata_rows(&self) -> Vec<(RngStreamRole, String, String)> {
        RngStreamRole::ALL
            .iter()
            .map(|&role| {
                let s = self.get(role);
                (role, s.coupling_key().to_owned(), s.seed_hex())
            })
            .collect()
    }
}
