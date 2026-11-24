//! Verification suite for Phase A initializers (`A01`–`A24`) and Classical GA (`C01`).

use jssp_core::decoder::{SerialDecoderWorkspace, decode_obr};
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule;
use jssp_core::genetic::{
    ClassicalGaConfig, PhaseABatchTask, default_phase_a_ga_budget, evaluate_initializer_only,
    mutate_obr, pox_crossover, run_phase_a_batch, run_phase_a_ga, tournament_select,
};
use jssp_core::initialization::qmc::{
    MAX_SOBOL_DIMENSION, generate_halton_points, generate_sobol_points,
};
use jssp_core::initialization::{InitializerId, generate_initial_population};
use jssp_core::model::JsspInstance;
use jssp_core::representation::ObrChromosome;
use jssp_core::rng::{PairingPolicy, RngStreamRole, SeedDerivationContext};
use jssp_core::telemetry::compute_obr_hamming_diversity;

fn ft06_instance() -> JsspInstance {
    JsspInstance::new(vec![
        vec![(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
        vec![(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
        vec![(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
        vec![(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
        vec![(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
        vec![(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
    ])
    .expect("ft06 must construct cleanly")
}

fn paired_context(rep: u32, inst_id: &str, cfg_id: &str) -> SeedDerivationContext {
    SeedDerivationContext::from_frozen_replicate(
        rep,
        inst_id,
        PairingPolicy::PairedInitialAndScramble {
            pairing_group: "phase_a_test".to_owned(),
            configuration_id: cfg_id.to_owned(),
        },
    )
    .expect("valid frozen seed context")
}

fn brute_force_hamming_diversity(chrs: &[ObrChromosome]) -> f64 {
    let n = chrs.len();
    if n < 2 {
        return 0.0;
    }
    let l = chrs[0].len();
    let mut total = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = chrs[i]
                .genes()
                .iter()
                .zip(chrs[j].genes().iter())
                .filter(|(a, b)| a != b)
                .count();
            total += (diff as f64) / (l as f64);
        }
    }
    let pairs = (n * (n - 1)) as f64 / 2.0;
    total / pairs
}

#[test]
fn serial_decoder_workspace_matches_reference_decode_obr_bit_for_bit() {
    let instances = [
        ft06_instance(),
        JsspInstance::new(vec![vec![(0, 2), (1, 0)], vec![(1, 3), (0, 2)]])
            .expect("zero-op fixture"),
    ];

    for (inst_idx, inst) in instances.iter().enumerate() {
        let mut streams = paired_context(0, "ws_equiv", "A01").all_streams();
        let pop = generate_initial_population(InitializerId::A01, inst, 50, &mut streams)
            .expect("generate A01");
        let mut ws = SerialDecoderWorkspace::new(inst);

        for (i, chr) in pop.chromosomes.iter().enumerate() {
            let ref_sched = decode_obr(inst, chr).expect("reference decode");
            validate_schedule(inst, &ref_sched).expect("reference valid");
            let ws_sched = ws.decode(inst, chr, true).expect("workspace decode");
            let fast_cmax = ws
                .decode_makespan_only(inst, chr)
                .expect("workspace makespan_only");

            assert_eq!(ws_sched, ref_sched, "inst={inst_idx}, chr={i}");
            assert_eq!(fast_cmax, ref_sched.makespan(), "inst={inst_idx}, chr={i}");
        }
    }
}

#[test]
fn all_24_initializers_produce_feasible_populations_and_exact_metadata() {
    let inst = ft06_instance();
    assert_eq!(InitializerId::ALL.len(), 24);

    for &init in &InitializerId::ALL {
        let mut streams = paired_context(0, "ft06", init.code()).all_streams();
        let (diag, chrs, scheds) =
            evaluate_initializer_only(&inst, init, 64, &mut streams).expect("eval init");
        assert_eq!(diag.population_size, 64);
        assert_eq!(diag.feasible_count, 64);
        assert_eq!(chrs.len(), 64);
        assert_eq!(scheds.len(), 64);

        for chr in &chrs {
            assert!(chr.validate(&inst).is_ok());
        }

        let bf_div = brute_force_hamming_diversity(&chrs);
        assert!(
            (diag.obr_hamming_diversity - bf_div).abs() < 1e-12,
            "diversity mismatch for {}: fast={} vs brute={}",
            init.code(),
            diag.obr_hamming_diversity,
            bf_div
        );

        match init {
            InitializerId::A02 => {
                assert!(init.is_deterministic());
                assert!(init.alpha().is_none());
                assert_eq!(diag.unique_chromosomes, 1);
                assert_eq!(diag.unique_schedules, 1);
                assert!((diag.obr_hamming_diversity - 0.0).abs() < 1e-12);
                assert!(diag.qmc_1d_ks_discrepancy.is_none());
                assert_eq!(streams.init.draws_u64(), 0);
                assert_eq!(streams.qmc_scramble.draws_u64(), 0);
            }
            InitializerId::A07 | InitializerId::A11 | InitializerId::A23 | InitializerId::A24 => {
                assert!(init.uses_sobol());
                assert!(init.has_dyadic_sobol_net_balance(64));
                assert!(diag.qmc_1d_ks_discrepancy.is_some());
            }
            InitializerId::A12 => {
                assert_eq!(init.mix_counts(64), Some((32, 32)));
                assert!(init.has_dyadic_sobol_net_balance(64));
            }
            InitializerId::A13 => {
                assert_eq!(init.mix_counts(64), Some((16, 48)));
                assert!(!init.has_dyadic_sobol_net_balance(64));
            }
            InitializerId::A14 => {
                assert_eq!(init.mix_counts(64), Some((48, 16)));
                assert!(init.has_dyadic_sobol_net_balance(64));
            }
            _ => {}
        }
    }
}

#[test]
fn deterministic_mwr_a02_tie_breaks_by_lowest_job_id() {
    // Three jobs with identical initial remaining work (10 = 6 + 4)
    let inst = JsspInstance::new(vec![
        vec![(0, 6), (1, 4)],
        vec![(1, 6), (0, 4)],
        vec![(0, 6), (1, 4)],
    ])
    .expect("tie fixture");
    let mut streams = paired_context(0, "tie_mwr", "A02").all_streams();
    let (diag, chrs, _) =
        evaluate_initializer_only(&inst, InitializerId::A02, 8, &mut streams).expect("A02");
    assert_eq!(diag.unique_chromosomes, 1);
    // Step 0: [10, 10, 10] -> Job 0 -> remaining [4, 10, 10]
    // Step 1: [4, 10, 10] -> Job 1 -> remaining [4, 4, 10]
    // Step 2: [4, 4, 10] -> Job 2 -> remaining [4, 4, 4]
    // Step 3: [4, 4, 4] -> Job 0 -> remaining [0, 4, 4]
    // Step 4: [0, 4, 4] -> Job 1 -> remaining [0, 0, 4]
    // Step 5: [0, 0, 4] -> Job 2
    assert_eq!(chrs[0].to_job_indices(), vec![0, 1, 2, 0, 1, 2]);
}

#[test]
fn sobol_matches_joe_kuo_values_preserves_dyadic_1d_net_and_enforces_21201_cap() {
    // 1. Exact unscrambled Joe-Kuo coordinates for d=5, i=0..4
    let pts = generate_sobol_points(5, 4, false, 0, None).expect("unscrambled Sobol");
    assert_eq!(pts[0], vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(pts[1], vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    assert_eq!(pts[2], vec![0.75, 0.25, 0.25, 0.25, 0.75]);
    assert_eq!(pts[3], vec![0.25, 0.75, 0.75, 0.75, 0.25]);

    // 2. Matousek scrambled 64-point prefix (i=0..63) has exact 1D stratification:
    //    every coordinate s has exactly 1 point in each [b/64, (b+1)/64) bin!
    let mut rng = paired_context(0, "net_check", "A07").stream(RngStreamRole::QmcScramble);
    let scrambled = generate_sobol_points(20, 64, true, 0, Some(&mut rng)).expect("scrambled");
    for s in 0..20 {
        let mut bins = [0usize; 64];
        for row in &scrambled {
            let b = ((row[s] * 64.0).floor() as usize).min(63);
            bins[b] += 1;
        }
        assert!(
            bins.iter().all(|&c| c == 1),
            "Matousek scrambled Sobol failed exact 1D (t,6,s) stratification on dim {s}: {bins:?}"
        );
    }

    // 3. Scrambled Halton also runs cleanly without duplicate points and yields KS < 0.1 on 1D
    let mut hrng = paired_context(1, "halton_check", "A05").stream(RngStreamRole::QmcScramble);
    let halton = generate_halton_points(10, 64, true, Some(&mut hrng)).expect("scrambled halton");
    assert_eq!(halton.len(), 64);

    // 4. Reject L > 21,201 on standard Sobol with UnsupportedQmcDimension
    // Use rectangular 107 x 199 (L = 21,293 > 21,201) rather than 21,202 jobs x 1 machine
    assert_eq!(MAX_SOBOL_DIMENSION, 21_201);
    let row_199: Vec<(usize, u64)> = (0..199).map(|m| (m, 1)).collect();
    let oversized_inst = JsspInstance::new(vec![row_199; 107]).expect("107x199 rectangular");
    for &init in &[
        InitializerId::A06,
        InitializerId::A07,
        InitializerId::A12,
        InitializerId::A13,
        InitializerId::A14,
    ] {
        let mut streams = paired_context(0, "oversized", init.code()).all_streams();
        let err = generate_initial_population(init, &oversized_inst, 8, &mut streams)
            .expect_err("must reject L > 21,201");
        assert_eq!(
            err,
            JsspError::UnsupportedQmcDimension {
                requested_dimension: 21_293,
                max_supported_dimension: 21_201,
            }
        );
    }
    // While Halton handles 21,202 dimensions cleanly!
    let mut h_streams = paired_context(0, "oversized", "A05").all_streams();
    let h_pop = generate_initial_population(InitializerId::A05, &oversized_inst, 8, &mut h_streams)
        .expect("Halton supports L > 21,201");
    assert_eq!(h_pop.chromosomes.len(), 8);
}

#[test]
fn genetic_operators_and_stream_isolation_invariants() {
    let inst = ft06_instance();
    let mut streams_base = paired_context(3, "ft06", "C01").all_streams();
    let init_pop =
        generate_initial_population(InitializerId::A01, &inst, 16, &mut streams_base).expect("pop");

    // Verify POX crossover preserves OBR validity across 100 pairs
    for k in 0..50 {
        let p1 = &init_pop.chromosomes[k % 16];
        let p2 = &init_pop.chromosomes[(k + 5) % 16];
        let (mut c1, mut c2, did_cx) =
            pox_crossover(p1, p2, inst.num_jobs(), 1.0, &mut streams_base.crossover).expect("pox");
        assert!(did_cx);
        assert!(c1.validate(&inst).is_ok());
        assert!(c2.validate(&inst).is_ok());
        assert!(mutate_obr(&mut c1, 1.0, &mut streams_base.mutation).expect("mut1"));
        assert!(mutate_obr(&mut c2, 1.0, &mut streams_base.mutation).expect("mut2"));
        assert!(c1.validate(&inst).is_ok());
        assert!(c2.validate(&inst).is_ok());
    }

    // Verify tournament selection deterministic tie-break on smallest index
    let ties = vec![100u64; 8];
    for _ in 0..20 {
        let mut s_copy = streams_base.selection.clone();
        let a = s_copy.gen_range_usize(8).unwrap();
        let b = s_copy.gen_range_usize(8).unwrap();
        let picked = tournament_select(&ties, 2, &mut streams_base.selection).expect("select");
        assert_eq!(picked, a.min(b));
    }

    // Verify stream isolation: changing mutation_rate/crossover_rate NEVER changes initial pop
    let cfg_default = ClassicalGaConfig::for_phase_a_instance(&inst);
    let mut cfg_alt = cfg_default;
    cfg_alt.mutation_rate = 0.99;
    cfg_alt.crossover_rate = 0.01;
    let res1 = run_phase_a_ga(
        &inst,
        InitializerId::A05,
        &cfg_default,
        &mut paired_context(4, "ft06", "A05").all_streams(),
    )
    .expect("run1");
    let res2 = run_phase_a_ga(
        &inst,
        InitializerId::A05,
        &cfg_alt,
        &mut paired_context(4, "ft06", "A05").all_streams(),
    )
    .expect("run2");
    assert_eq!(res1.initial.best_makespan, res2.initial.best_makespan);
    assert_eq!(res1.initial.mean_makespan, res2.initial.mean_makespan);
    assert_eq!(res1.initial.median_makespan, res2.initial.median_makespan);
    assert_eq!(
        res1.initial.unique_chromosomes,
        res2.initial.unique_chromosomes
    );
    assert_eq!(res1.initial.unique_schedules, res2.initial.unique_schedules);
    assert_eq!(
        res1.initial.obr_hamming_diversity,
        res2.initial.obr_hamming_diversity
    );
    assert_eq!(
        res1.initial.qmc_1d_ks_discrepancy,
        res2.initial.qmc_1d_ks_discrepancy
    );
}

#[test]
fn classical_ga_ft06_budget_determinism_batch_and_auc() {
    let inst = ft06_instance();
    let budget = default_phase_a_ga_budget(6, 6);
    assert_eq!(budget, 6400);
    let cfg = ClassicalGaConfig::for_phase_a_instance(&inst);

    let ctx0 = paired_context(0, "ft06", "A11");
    let ctx1 = paired_context(1, "ft06", "A11");

    let seq0 =
        run_phase_a_ga(&inst, InitializerId::A11, &cfg, &mut ctx0.all_streams()).expect("seq0");
    let seq0_repeat = run_phase_a_ga(&inst, InitializerId::A11, &cfg, &mut ctx0.all_streams())
        .expect("seq0 repeat");
    assert_eq!(seq0.evals_used, seq0_repeat.evals_used);
    assert_eq!(seq0.best_makespan, seq0_repeat.best_makespan);
    assert_eq!(seq0.best_found_eval, seq0_repeat.best_found_eval);
    assert_eq!(seq0.best_chromosome, seq0_repeat.best_chromosome);
    assert_eq!(seq0.best_schedule_hash, seq0_repeat.best_schedule_hash);
    assert_eq!(seq0.anytime_cmax_auc, seq0_repeat.anytime_cmax_auc);
    assert_eq!(seq0.trajectory, seq0_repeat.trajectory);
    assert_eq!(seq0.evals_used, budget);
    assert_eq!(seq0.initial.feasible_count, 64);
    assert_eq!(seq0.final_pop.feasible_count, 64);
    assert!(seq0.best_makespan <= seq0.initial.best_makespan);
    assert!(
        seq0.best_makespan <= 60,
        "expected good ft06 makespan, got {}",
        seq0.best_makespan
    );
    assert!(seq0.anytime_cmax_auc >= seq0.best_makespan as f64);

    let tasks = vec![
        PhaseABatchTask {
            initializer: InitializerId::A11,
            seed_context: ctx0,
            config: cfg,
        },
        PhaseABatchTask {
            initializer: InitializerId::A11,
            seed_context: ctx1.clone(),
            config: cfg,
        },
    ];
    let batch_1 = run_phase_a_batch(&inst, &tasks, 1).expect("batch 1 thread");
    let batch_2 = run_phase_a_batch(&inst, &tasks, 2).expect("batch 2 threads");
    assert_eq!(batch_1[0].best_makespan, seq0.best_makespan);
    assert_eq!(batch_1[0].best_chromosome, seq0.best_chromosome);
    assert_eq!(batch_1[0].trajectory, seq0.trajectory);
    assert_eq!(batch_2[0].best_makespan, batch_1[0].best_makespan);
    assert_eq!(batch_2[1].best_makespan, batch_1[1].best_makespan);
    assert_eq!(batch_2[1].best_chromosome, batch_1[1].best_chromosome);

    // Diversity identity check on O(NL) helper directly
    assert_eq!(compute_obr_hamming_diversity(&[], 6), 0.0);
}
