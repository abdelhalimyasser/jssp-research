//! Deterministic RNG stream derivation, pairing policy, and primitive reference tests.

use std::collections::HashSet;

use jssp_core::error::JsspError;
use jssp_core::rng::{
    FROZEN_MASTER_SEEDS_V1, PairingPolicy, RNG_ALGORITHM_ID, RNG_CRATE_VERSION, RngStreamRole,
    SeedDerivationContext, compute_frozen_master_seed, master_seed_for_replicate,
};

fn base_paired_context(config_id: &str) -> SeedDerivationContext {
    SeedDerivationContext::from_frozen_replicate(
        0,
        "ft06",
        PairingPolicy::PairedInitialAndScramble {
            pairing_group: "phase_b_shared".to_owned(),
            configuration_id: config_id.to_owned(),
        },
    )
    .expect("valid frozen replicate 0")
}

#[test]
fn frozen_master_seed_table_matches_sha256_definition() {
    assert_eq!(RNG_ALGORITHM_ID, "ChaCha20Rng");
    assert_eq!(RNG_CRATE_VERSION, "rand_chacha-0.10.0");

    for r in 0..20u32 {
        let computed = compute_frozen_master_seed(r).expect("valid replicate index");
        let lookup = master_seed_for_replicate(r).expect("valid replicate lookup");
        assert_eq!(computed, lookup);
        assert_eq!(computed, FROZEN_MASTER_SEEDS_V1[r as usize]);
    }

    assert_eq!(
        master_seed_for_replicate(20),
        Err(JsspError::InvalidReplicateIndex {
            replicate_index: 20,
            max_replicates: 20
        })
    );
}

#[test]
fn same_context_reproduces_identical_outputs_and_differs_across_roles() {
    let ctx_a = base_paired_context("B01");
    let ctx_b = base_paired_context("B01");

    let mut bundle_a = ctx_a.all_streams();
    let mut bundle_b = ctx_b.all_streams();

    let mut unique_seeds = HashSet::new();
    for role in RngStreamRole::ALL {
        let stream_a = bundle_a.get_mut(role);
        let stream_b = bundle_b.get_mut(role);
        assert_eq!(stream_a.seed_bytes(), stream_b.seed_bytes());
        assert_eq!(stream_a.coupling_key(), stream_b.coupling_key());
        assert!(
            unique_seeds.insert(*stream_a.seed_bytes()),
            "all 9 roles must derive distinct 256-bit seed material"
        );

        let draws_a: Vec<u64> = (0..16).map(|_| stream_a.next_u64()).collect();
        let draws_b: Vec<u64> = (0..16).map(|_| stream_b.next_u64()).collect();
        assert_eq!(draws_a, draws_b);
    }
    assert_eq!(unique_seeds.len(), 9);
}

#[test]
fn stream_construction_order_and_consumption_do_not_interfere() {
    let ctx = base_paired_context("C06");

    // Derive directly in reverse order and consume heavily on `Mutation` and `Init`
    let mut direct_crossover = ctx.stream(RngStreamRole::Crossover);
    let mut direct_mutation = ctx.stream(RngStreamRole::Mutation);
    let mut direct_init = ctx.stream(RngStreamRole::Init);

    for _ in 0..10_000 {
        let _ = direct_init.next_u64();
        let _ = direct_mutation.next_u64();
    }

    let bundle = ctx.all_streams();
    let mut bundled_crossover = bundle.crossover;

    let seq_direct: Vec<u64> = (0..32).map(|_| direct_crossover.next_u64()).collect();
    let seq_bundled: Vec<u64> = (0..32).map(|_| bundled_crossover.next_u64()).collect();
    assert_eq!(seq_direct, seq_bundled);
    assert_eq!(direct_init.draws_u64(), 10_000);
    assert_eq!(bundle.sa_accept.draws_u64(), 0);
}

#[test]
fn pairing_policies_govern_shared_vs_configuration_specific_streams() {
    let ctx_b01 = base_paired_context("B01");
    let ctx_b02 = base_paired_context("B02");

    // `init` and `qmc_scramble` must be paired identically across B01 and B02
    assert_eq!(
        ctx_b01.derive_seed(RngStreamRole::Init),
        ctx_b02.derive_seed(RngStreamRole::Init)
    );
    assert_eq!(
        ctx_b01.derive_seed(RngStreamRole::QmcScramble),
        ctx_b02.derive_seed(RngStreamRole::QmcScramble)
    );

    // Search/operator streams must be isolated between B01 and B02
    assert_ne!(
        ctx_b01.derive_seed(RngStreamRole::LocalSearchSample),
        ctx_b02.derive_seed(RngStreamRole::LocalSearchSample)
    );
    assert_ne!(
        ctx_b01.derive_seed(RngStreamRole::Perturbation),
        ctx_b02.derive_seed(RngStreamRole::Perturbation)
    );
    assert_ne!(
        ctx_b01.derive_seed(RngStreamRole::MhFilter),
        ctx_b02.derive_seed(RngStreamRole::MhFilter)
    );

    // SharedStreamSeeds shares identical initial 256-bit ChaCha20 seeds across all 9 roles
    // (does not claim event alignment in search loops if configs draw different step counts).
    let shared = SeedDerivationContext::from_frozen_replicate(
        0,
        "ft06",
        PairingPolicy::SharedStreamSeeds {
            pairing_group: "phase_b_shared".to_owned(),
        },
    )
    .expect("shared_stream_seeds");
    assert_eq!(shared.pairing_policy.policy_name(), "shared_stream_seeds");
    for role in RngStreamRole::ALL {
        if role.is_paired_by_default() {
            assert_eq!(shared.derive_seed(role), ctx_b01.derive_seed(role));
        }
    }

    // Event-alignment distinction: paired_init gives identical initial OBR permutations on fixed
    // draw count L=9 even when downstream search operators draw different numbers of variates.
    let mut b01_streams = ctx_b01.all_streams();
    let mut b02_streams = ctx_b02.all_streams();
    let mut init_a = [0usize, 0, 1, 1, 2, 2];
    let mut init_b = [0usize, 0, 1, 1, 2, 2];
    b01_streams.init.shuffle(&mut init_a);
    b02_streams.init.shuffle(&mut init_b);
    assert_eq!(init_a, init_b);

    // Downstream search operators differ in draw schedules and seeds under paired_init
    let draw_b01_ls = b01_streams.local_search_sample.next_u64();
    let draw_b02_ls = b02_streams.local_search_sample.next_u64();
    assert_ne!(draw_b01_ls, draw_b02_ls);

    // Even under SharedStreamSeeds, diverging draw counts shift stream cursor alignment
    let mut s_stream_a = shared.stream(RngStreamRole::LocalSearchSample);
    let mut s_stream_b = shared.stream(RngStreamRole::LocalSearchSample);
    let _ = s_stream_a.next_u64();
    let _ = s_stream_a.next_u64();
    let _ = s_stream_b.next_u64();
    assert_ne!(s_stream_a.next_u64(), s_stream_b.next_u64());

    // Configuration-scoped isolates all streams when config IDs differ
    let scoped_a = SeedDerivationContext::from_frozen_replicate(
        0,
        "ft06",
        PairingPolicy::ConfigurationScoped {
            configuration_id: "A01".to_owned(),
        },
    )
    .expect("scoped a");
    let scoped_b = SeedDerivationContext::from_frozen_replicate(
        0,
        "ft06",
        PairingPolicy::ConfigurationScoped {
            configuration_id: "A03".to_owned(),
        },
    )
    .expect("scoped b");
    assert_ne!(
        scoped_a.derive_seed(RngStreamRole::Init),
        scoped_b.derive_seed(RngStreamRole::Init)
    );
}

#[test]
fn fixed_reference_vectors_for_raw_range_shuffle_and_sampling() {
    let ctx = SeedDerivationContext::with_explicit_master_seed(
        0x1234_5678_9ABC_DEF0,
        3,
        "abz5",
        PairingPolicy::ConfigurationScoped {
            configuration_id: "M3_REF_V1".to_owned(),
        },
    )
    .expect("valid reference context");

    let mut raw_stream = ctx.stream(RngStreamRole::Init);
    assert_eq!(
        raw_stream.seed_hex(),
        "ea9abd3c2ced43d77e88dea00ffeee1caf3e4102a922fced131a7aaa07798936"
    );

    let words: Vec<u64> = (0..4).map(|_| raw_stream.next_u64()).collect();
    assert_eq!(
        words,
        vec![
            0xEAE7_4E19_436C_454D,
            0x8D9D_878B_257C_44FA,
            0xC4A6_C877_C0FF_87D1,
            0xCE7D_D86E_A467_2E64,
        ]
    );

    let mut range_stream = ctx.stream(RngStreamRole::Init);
    let range_vals: Vec<u64> = (0..8)
        .map(|_| range_stream.gen_range_u64(10).expect("upper > 0"))
        .collect();
    assert_eq!(range_vals, vec![9, 6, 7, 0, 0, 7, 1, 1]);
    assert_eq!(
        range_stream.gen_range_u64(0),
        Err(JsspError::InvalidRandomBound { upper: 0 })
    );
    assert_eq!(
        range_stream.choose_index(0),
        Err(JsspError::InvalidRandomBound { upper: 0 })
    );

    let mut shuffle_stream = ctx.stream(RngStreamRole::Init);
    let mut items: Vec<usize> = (0..10).collect();
    shuffle_stream.shuffle(&mut items);
    assert_eq!(items, vec![7, 5, 0, 3, 2, 6, 8, 1, 4, 9]);

    let mut sample_stream = ctx.stream(RngStreamRole::Init);
    let sampled = sample_stream
        .sample_indices_without_replacement(10, 5)
        .expect("k <= n");
    assert_eq!(sampled.len(), 5);
    assert_eq!(sampled.iter().copied().collect::<HashSet<usize>>().len(), 5);
    let mut sample_stream_repeat = ctx.stream(RngStreamRole::Init);
    assert_eq!(
        sampled,
        sample_stream_repeat
            .sample_indices_without_replacement(10, 5)
            .expect("repeat deterministic")
    );
    assert_eq!(
        sample_stream.sample_indices_without_replacement(4, 5),
        Err(JsspError::InvalidSampleSize {
            population_len: 4,
            requested: 5
        })
    );
}
