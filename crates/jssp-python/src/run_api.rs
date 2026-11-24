//! PyO3 coarse-grained decoding and OBR resolution entrypoints.

use jssp_core::decoder::{
    decode_obr as core_decode_obr, decode_obr_append_only as core_decode_obr_append_only,
};
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule as core_validate_schedule;
use jssp_core::representation::ObrChromosome;
use jssp_core::rng::{
    CHACHA_ROUNDS, FROZEN_MASTER_SEEDS_V1, HASH_CRATE_VERSION, MASTER_SEED_SCHEDULE_ID_V1,
    NamedRngStream, PairingPolicy, RNG_ALGORITHM_ID, RNG_CRATE_VERSION, RngStreamRole,
    SEED_DERIVATION_SCHEMA_V1, SeedDerivationContext, hex_to_seed, master_seed_for_replicate,
};
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::result_api::{DecodedSchedulePayload, schedule_to_py_payload};

/// Decodes a validated OBR chromosome with the canonical deterministic Serial SGS
/// earliest feasible idle-gap insertion decoder and independently validates the result.
#[pyfunction]
pub fn decode_obr(
    jobs: Vec<Vec<(usize, u64)>>,
    chromosome: Vec<usize>,
) -> PyResult<DecodedSchedulePayload> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    let schedule = core_decode_obr(&instance, &obr).map_err(jssp_err_to_py)?;
    core_validate_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    Ok(schedule_to_py_payload(&schedule))
}

/// Decodes an OBR chromosome using the semi-active append-only reference decoder
/// (for comparative verification against [`decode_obr`]).
#[pyfunction]
pub fn decode_obr_append_only(
    jobs: Vec<Vec<(usize, u64)>>,
    chromosome: Vec<usize>,
) -> PyResult<DecodedSchedulePayload> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    let schedule = core_decode_obr_append_only(&instance, &obr).map_err(jssp_err_to_py)?;
    core_validate_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    Ok(schedule_to_py_payload(&schedule))
}

/// Validates an OBR chromosome and resolves each position into `(job, op_index)`.
#[pyfunction]
pub fn resolve_obr_operations(
    jobs: Vec<Vec<(usize, u64)>>,
    chromosome: Vec<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    let ops = obr.resolve_operations(&instance).map_err(jssp_err_to_py)?;
    Ok(ops
        .into_iter()
        .map(|op_id| (op_id.job.0, op_id.op_index.0))
        .collect())
}

type RngMetadataTuple = (
    &'static str,
    &'static str,
    &'static str,
    u32,
    &'static str,
    &'static str,
    Vec<&'static str>,
);

type DerivedStreamRow = (String, String, String, Vec<u64>);

pub(crate) fn build_pairing_policy(
    pairing_mode: &str,
    pairing_group: String,
    configuration_id: String,
) -> PyResult<PairingPolicy> {
    match pairing_mode {
        "shared_stream_seeds" | "crn" => {
            if pairing_group.is_empty() {
                return Err(jssp_err_to_py(JsspError::InvalidRngConfiguration {
                    detail: "pairing_group must not be empty for shared_stream_seeds policy"
                        .to_owned(),
                }));
            }
            Ok(PairingPolicy::SharedStreamSeeds { pairing_group })
        }
        "paired_init" => {
            if pairing_group.is_empty() || configuration_id.is_empty() {
                return Err(jssp_err_to_py(JsspError::InvalidRngConfiguration {
                    detail: "pairing_group and configuration_id must be non-empty for paired_init policy".to_owned(),
                }));
            }
            Ok(PairingPolicy::PairedInitialAndScramble {
                pairing_group,
                configuration_id,
            })
        }
        "config_scoped" => {
            if configuration_id.is_empty() {
                return Err(jssp_err_to_py(JsspError::InvalidRngConfiguration {
                    detail: "configuration_id must not be empty for config_scoped policy"
                        .to_owned(),
                }));
            }
            Ok(PairingPolicy::ConfigurationScoped { configuration_id })
        }
        other => Err(jssp_err_to_py(JsspError::InvalidRngConfiguration {
            detail: format!(
                "unknown pairing_mode '{other}', expected 'shared_stream_seeds', 'paired_init', or 'config_scoped'"
            ),
        })),
    }
}

fn collect_stream_previews(
    context: &SeedDerivationContext,
    preview_count: usize,
) -> Vec<DerivedStreamRow> {
    let mut bundle = context.all_streams();
    RngStreamRole::ALL
        .iter()
        .map(|&role| {
            let stream = bundle.get_mut(role);
            let mut draws = Vec::with_capacity(preview_count);
            for _ in 0..preview_count {
                draws.push(stream.next_u64());
            }
            (
                role.as_str().to_owned(),
                stream.coupling_key().to_owned(),
                stream.seed_hex(),
                draws,
            )
        })
        .collect()
}

/// Returns the frozen 20-element confirmatory master seed array (`r = 0..20`).
#[pyfunction]
pub fn frozen_master_seeds() -> Vec<u64> {
    FROZEN_MASTER_SEEDS_V1.to_vec()
}

/// Returns authoritative Rust RNG metadata `(algorithm, crate_version, hash_crate, rounds, schema, schedule_id, roles)`.
#[pyfunction]
pub fn rng_metadata() -> RngMetadataTuple {
    (
        RNG_ALGORITHM_ID,
        RNG_CRATE_VERSION,
        HASH_CRATE_VERSION,
        CHACHA_ROUNDS,
        SEED_DERIVATION_SCHEMA_V1,
        MASTER_SEED_SCHEDULE_ID_V1,
        RngStreamRole::ALL.iter().map(|r| r.as_str()).collect(),
    )
}

/// Derives all nine named RNG streams using the frozen confirmatory master seed for `replicate_index`.
#[pyfunction]
pub fn derive_rng_streams(
    replicate_index: u32,
    instance_id: String,
    pairing_mode: String,
    pairing_group: String,
    configuration_id: String,
    preview_count: usize,
) -> PyResult<Vec<DerivedStreamRow>> {
    let master_seed = master_seed_for_replicate(replicate_index).map_err(jssp_err_to_py)?;
    derive_rng_streams_with_seed(
        master_seed,
        replicate_index,
        instance_id,
        pairing_mode,
        pairing_group,
        configuration_id,
        preview_count,
    )
}

/// Derives all nine named RNG streams using an explicit 64-bit `master_seed`.
#[pyfunction]
pub fn derive_rng_streams_with_seed(
    master_seed: u64,
    replicate_index: u32,
    instance_id: String,
    pairing_mode: String,
    pairing_group: String,
    configuration_id: String,
    preview_count: usize,
) -> PyResult<Vec<DerivedStreamRow>> {
    let policy = build_pairing_policy(&pairing_mode, pairing_group, configuration_id)?;
    let context = SeedDerivationContext::with_explicit_master_seed(
        master_seed,
        replicate_index,
        instance_id,
        policy,
    )
    .map_err(jssp_err_to_py)?;
    Ok(collect_stream_previews(&context, preview_count))
}

/// Samples `count` uniform integers in `[0, upper)` and shuffles `0..shuffle_len`
/// sequentially using a `ChaCha20Rng` initialized from a 64-char hex 256-bit seed.
#[pyfunction]
pub fn rng_sample_range_and_shuffle(
    seed_hex: String,
    upper: u64,
    count: usize,
    shuffle_len: usize,
) -> PyResult<(Vec<u64>, Vec<usize>)> {
    let seed_bytes = hex_to_seed(&seed_hex).map_err(jssp_err_to_py)?;
    let mut stream = NamedRngStream::from_raw_seed(RngStreamRole::Init, seed_bytes, "direct");

    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        samples.push(stream.gen_range_u64(upper).map_err(jssp_err_to_py)?);
    }

    let mut permutation: Vec<usize> = (0..shuffle_len).collect();
    stream.shuffle(&mut permutation);
    Ok((samples, permutation))
}
