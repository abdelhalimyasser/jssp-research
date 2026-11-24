//! Reproducible named ChaCha20 PRNG substreams and unbiased sampling primitives.

pub mod streams;

pub use streams::{
    ARITHMETIC_FALLBACK_SEEDS_V1, CHACHA_ROUNDS, FROZEN_MASTER_SEEDS_V1, HASH_CRATE_VERSION,
    MASTER_SEED_SCHEDULE_ID_V1, NUM_CONFIRMATORY_SEEDS, NamedRngStream, PairingPolicy,
    RNG_ALGORITHM_ID, RNG_CRATE_VERSION, RngStreamRole, RunRngStreams, SEED_DERIVATION_SCHEMA_V1,
    SeedDerivationContext, compute_frozen_master_seed, derive_stream_seed_bytes, hex_to_seed,
    master_seed_for_replicate, seed_to_hex,
};
