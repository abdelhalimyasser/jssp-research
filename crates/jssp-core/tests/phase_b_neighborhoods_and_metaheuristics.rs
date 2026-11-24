//! Verification suite for Phase B neighborhoods (`N5`–`N8`), exact DAG evaluation (`D03-C`),
//! D04-B 3-cycle perturbation, and all 28 active metaheuristics (`B01`–`B24`, `B29`–`B32`).

use std::collections::{BTreeSet, HashMap, HashSet};

use jssp_core::decoder::{SerialDecoderWorkspace, decode_obr};
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule;
use jssp_core::graph::analyze_schedule;
use jssp_core::initialization::{InitializerId, generate_initial_population};
use jssp_core::local_search::descent::{SearchTerminationReason, compute_stepwise_anytime_auc};
use jssp_core::local_search::evaluation::FastGraphEvaluator;
use jssp_core::metaheuristics::{
    PhaseBBatchTask, PhaseBConfigId, default_phase_b_budget, generate_matched_phase_b_start,
    perturb_obr_three_cycle, run_phase_b_batch, run_phase_b_from_schedule,
};
use jssp_core::model::{
    JobId, JsspInstance, MachineId, MachineSequence, OperationId, OperationIndex,
};
use jssp_core::neighborhoods::{NeighborhoodId, apply_move_to_slice, generate_neighborhood};
use jssp_core::representation::{ObrChromosome, round_trip_machine_sequences};
use jssp_core::rng::{PairingPolicy, SeedDerivationContext};

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

fn zero_duration_instance() -> JsspInstance {
    JsspInstance::new(vec![
        vec![(0, 4), (1, 0), (2, 3)],
        vec![(1, 5), (2, 2), (0, 4)],
        vec![(2, 6), (0, 2), (1, 3)],
    ])
    .expect("zero-duration fixture must construct")
}

fn paired_phase_b_context(rep: u32, inst_id: &str, cfg_id: &str) -> SeedDerivationContext {
    SeedDerivationContext::from_frozen_replicate(
        rep,
        inst_id,
        PairingPolicy::PairedInitialAndScramble {
            pairing_group: "phase_b_matched_start".to_owned(),
            configuration_id: cfg_id.to_owned(),
        },
    )
    .expect("valid frozen context")
}

fn resulting_seq_signature(seq_vecs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    seq_vecs.to_vec()
}

#[test]
fn fast_graph_evaluator_matches_m4_disjunctive_graph_on_50_schedules_and_zero_duration() {
    let instances = [ft06_instance(), zero_duration_instance()];

    for (idx, inst) in instances.iter().enumerate() {
        let mut streams = paired_phase_b_context(0, "diff_m4", "A01").all_streams();
        let count = if idx == 0 { 50 } else { 12 };
        let pop = generate_initial_population(InitializerId::A01, inst, count, &mut streams)
            .expect("generate A01");
        let mut eval = FastGraphEvaluator::new(inst).expect("evaluator");
        let mut ws = SerialDecoderWorkspace::new(inst);
        let nm = inst.num_machines();

        for chr in &pop.chromosomes {
            let sched = decode_obr(inst, chr).expect("decode");
            let (_, m4) = analyze_schedule(inst, &sched).expect("M4 analyze_schedule");
            eval.load_from_schedule(&sched).expect("load_from_schedule");
            let fast = eval.analyze_current().expect("dp ok").expect("acyclic");

            assert_eq!(fast.makespan, m4.graph_makespan());
            for (u, op_crit) in m4.operations().iter().enumerate() {
                assert_eq!(fast.heads[u], op_crit.head, "head mismatch at u={u}");
                assert_eq!(fast.tails[u], op_crit.tail, "tail mismatch at u={u}");
            }

            let m4_path_flat: Vec<u32> = m4
                .representative_critical_path()
                .operations
                .iter()
                .map(|op| op.flat_index(inst.num_jobs(), nm).expect("flat") as u32)
                .collect();
            assert_eq!(fast.critical_path, m4_path_flat);

            // Partition check: concatenating fast.blocks reproduces representative critical_path
            let mut recon_path = Vec::new();
            let mut len2_blocks_ops = Vec::new();
            for (bi, blk) in fast.blocks.iter().enumerate() {
                assert_eq!(blk.block_index, bi);
                assert_eq!(blk.total_blocks, fast.blocks.len());
                assert!(!blk.operations.is_empty());
                if bi > 0 {
                    assert_ne!(blk.machine, fast.blocks[bi - 1].machine);
                }
                recon_path.extend_from_slice(&blk.operations);
                if blk.operations.len() >= 2 {
                    len2_blocks_ops.push((blk.machine, blk.operations.clone()));
                }
            }
            assert_eq!(recon_path, fast.critical_path);

            let m4_blocks_ops: Vec<(usize, Vec<u32>)> = m4
                .critical_blocks()
                .iter()
                .map(|b| {
                    let ops = b
                        .operations
                        .iter()
                        .map(|op| op.flat_index(inst.num_jobs(), nm).expect("flat") as u32)
                        .collect();
                    (b.machine.0, ops)
                })
                .collect();
            for &(m, ref ops) in &len2_blocks_ops {
                assert!(
                    m4_blocks_ops.iter().any(|(mm, mops)| *mm == m
                        && mops.windows(ops.len()).any(|w| w == ops.as_slice())),
                    "Representative critical block ({m}, {ops:?}) must lie inside M4 critical subgraph blocks {m4_blocks_ops:?}"
                );
            }

            // Round-trip projection check via D01-B
            let (proj_obr, proj_sched, _shifts) =
                eval.project_and_validate(inst, &mut ws).expect("project");
            proj_obr.validate(inst).expect("valid obr");
            validate_schedule(inst, &proj_sched).expect("valid schedule");
            assert!(proj_sched.makespan() <= fast.makespan);
        }
    }
}

#[test]
fn neighborhood_inclusion_dedup_cycle_rollback_and_boundary_clipping() {
    let inst = ft06_instance();
    let mut streams = paired_phase_b_context(1, "incl_ft06", "A01").all_streams();
    let pop = generate_initial_population(InitializerId::A01, &inst, 30, &mut streams)
        .expect("generate A01");
    let mut eval = FastGraphEvaluator::new(&inst).expect("evaluator");

    let mut observed_len1_end_clip = false;

    for chr in &pop.chromosomes {
        let sched = decode_obr(&inst, chr).expect("decode");
        eval.load_from_schedule(&sched).expect("load");
        let state = eval.analyze_current().expect("ok").expect("acyclic");
        let base_seqs = eval.snapshot_machine_seqs();

        if let (Some(first_b), Some(last_b)) = (state.blocks.first(), state.blocks.last())
            && ((first_b.operations.len() == 1 && state.blocks.len() > 1)
                || (last_b.operations.len() == 1 && state.blocks.len() > 1))
        {
            observed_len1_end_clip = true;
        }

        let mut seq_sets: HashMap<NeighborhoodId, BTreeSet<Vec<Vec<u32>>>> = HashMap::new();
        for &nid in &NeighborhoodId::ALL {
            let gen_n = generate_neighborhood(nid, &eval, &state);
            let mut produced = BTreeSet::new();
            for mv in &gen_n.moves {
                let mut mutated = base_seqs.clone();
                apply_move_to_slice(&mut mutated[mv.machine], mv);
                let sig = resulting_seq_signature(&mutated);
                assert!(
                    produced.insert(sig),
                    "Duplicate resulting sequence in {:?} on move {mv:?}",
                    nid
                );

                // Verify probe_move_makespan always rolls back evaluator cleanly, even on cycles
                let probe_res = eval.probe_move_makespan(mv).expect("probe no err");
                assert_eq!(
                    eval.snapshot_machine_seqs(),
                    base_seqs,
                    "Evaluator machine_seqs mutated after probe!"
                );
                // Recompute reference via FastGraphEvaluator clone on mutated seqs to check exactness
                let mut check_eval = FastGraphEvaluator::new(&inst).expect("check_eval");
                check_eval.restore_machine_seqs(&mutated);
                let check_cmax = check_eval
                    .analyze_current()
                    .expect("check analyze")
                    .map(|s| s.makespan);
                assert_eq!(probe_res, check_cmax);
            }
            seq_sets.insert(nid, produced);
        }

        let s5 = &seq_sets[&NeighborhoodId::N5];
        let s6 = &seq_sets[&NeighborhoodId::N6];
        let s7 = &seq_sets[&NeighborhoodId::N7];
        let s8 = &seq_sets[&NeighborhoodId::N8];
        assert!(s5.is_subset(s6), "N5 must be a subset of N6");
        assert!(s6.is_subset(s7), "N6 must be a subset of N7");
        assert!(s7.is_subset(s8), "N7 must be a subset of N8");
    }

    assert!(
        observed_len1_end_clip,
        "Expected at least one ft06 schedule with len==1 boundary block"
    );
}

#[test]
fn b25_through_b28_rejected_per_d04_a_and_28_active_configs_valid() {
    assert_eq!(PhaseBConfigId::ALL.len(), 28);
    let codes: BTreeSet<&str> = PhaseBConfigId::ALL.iter().map(|c| c.code()).collect();
    assert_eq!(codes.len(), 28);

    for pruned in ["B25", "B26", "B27", "B28"] {
        let err = PhaseBConfigId::from_code(pruned).expect_err("pruned ID must fail");
        match err {
            JsspError::InvalidInitializerConfiguration { detail } => {
                assert!(detail.contains("D04-A"), "Unexpected detail: {detail}");
            }
            other => panic!("Unexpected error variant for {pruned}: {other:?}"),
        }
    }

    let inst = ft06_instance();
    assert_eq!(default_phase_b_budget(6, 6, 100), 3600);
    assert_eq!(default_phase_b_budget(6, 6, 500), 18_000);

    // Verify all 28 active configs produce exact matched starts from frozen A07, run safely,
    // preserve feasibility, and agree with M5 MachineSequence round-trip.
    let mut matched_init_cmax: Option<u64> = None;
    for &cfg in &PhaseBConfigId::ALL {
        let mut streams = paired_phase_b_context(0, "ft06", cfg.code()).all_streams();
        let (obr, start_sched) =
            generate_matched_phase_b_start(&inst, InitializerId::A07, 64, &mut streams)
                .expect("matched start");
        obr.validate(&inst).expect("valid start OBR");
        match matched_init_cmax {
            None => matched_init_cmax = Some(start_sched.makespan()),
            Some(expected) => assert_eq!(
                start_sched.makespan(),
                expected,
                "Matched start Cmax differed for {}",
                cfg.code()
            ),
        }

        let res = run_phase_b_from_schedule(&inst, cfg, &start_sched, 300, &mut streams)
            .expect("phase B run");
        assert_eq!(res.config_id, cfg.code());
        assert_eq!(res.evaluation_budget, 300);
        assert!(res.evals_used <= 300);
        assert!(res.feasible_final);
        assert!(res.best_makespan <= res.initial_makespan);
        assert_eq!(res.best_milestones[0], (0, res.initial_makespan));
        let expected_auc = compute_stepwise_anytime_auc(&res.best_milestones, 300);
        assert!((res.anytime_cmax_auc - expected_auc).abs() < 1e-12);

        // Verify returned best_machine_sequences via M5 adapter
        let typed_seqs: Vec<MachineSequence> = res
            .best_machine_sequences
            .iter()
            .enumerate()
            .map(|(m, seq)| MachineSequence {
                machine: MachineId(m),
                operations: seq
                    .iter()
                    .map(|&(j, op)| OperationId::new(JobId(j), OperationIndex(op)))
                    .collect(),
            })
            .collect();
        let bundle = round_trip_machine_sequences(&inst, &typed_seqs)
            .expect("best_machine_sequences must round-trip cleanly");
        assert_eq!(bundle.audit.target_graph_makespan, res.best_makespan);
        assert!(bundle.decoded_schedule.makespan() <= res.best_makespan);
    }
}

#[test]
fn descent_distinguishes_local_optimum_from_budget_exhausted() {
    let inst = ft06_instance();
    let mut s_start = paired_phase_b_context(2, "ft06_lo", "B01").all_streams();
    let (_, start_sched) =
        generate_matched_phase_b_start(&inst, InitializerId::A01, 16, &mut s_start).expect("start");

    // Exhaust budget early (budget = 2) -> BudgetExhausted (unless initial happened to have <=2 moves)
    let mut s_short = paired_phase_b_context(2, "ft06_lo", "B01").all_streams();
    let res_short =
        run_phase_b_from_schedule(&inst, PhaseBConfigId::B01, &start_sched, 2, &mut s_short)
            .expect("short descent");
    assert_eq!(
        res_short.terminated_reason,
        SearchTerminationReason::BudgetExhausted
    );
    assert!(!res_short.reached_local_optimum);
    assert_eq!(res_short.evals_used, 2);

    // Generous budget (5000) across Steepest (B01-B04) and First Improvement (B05-B08)
    let mut eval = FastGraphEvaluator::new(&inst).expect("eval");
    let nm = inst.num_machines();
    for cfg in [
        PhaseBConfigId::B01,
        PhaseBConfigId::B02,
        PhaseBConfigId::B03,
        PhaseBConfigId::B04,
        PhaseBConfigId::B05,
        PhaseBConfigId::B06,
        PhaseBConfigId::B07,
        PhaseBConfigId::B08,
    ] {
        let mut s_full = paired_phase_b_context(2, "ft06_lo", cfg.code()).all_streams();
        let res = run_phase_b_from_schedule(&inst, cfg, &start_sched, 5000, &mut s_full)
            .expect("full descent");
        assert_eq!(res.terminated_reason, SearchTerminationReason::LocalOptimum);
        assert!(res.reached_local_optimum);
        assert!(res.evals_used < 5000);

        // Independently verify no strictly improving acyclic move exists in best_machine_sequences
        let flat_seqs: Vec<Vec<u32>> = res
            .best_machine_sequences
            .iter()
            .map(|seq| seq.iter().map(|&(j, op)| (j * nm + op) as u32).collect())
            .collect();
        eval.restore_machine_seqs(&flat_seqs);
        let state = eval.analyze_current().expect("ok").expect("acyclic");
        let gen_n = generate_neighborhood(cfg.neighborhood(), &eval, &state);
        for mv in &gen_n.moves {
            if let Some(cand) = eval.probe_move_makespan(mv).expect("probe") {
                assert!(
                    cand >= state.makespan,
                    "Found unapplied improving move {mv:?} ({cand} < {}) at claimed LocalOptimum {}",
                    state.makespan,
                    cfg.code()
                );
            }
        }
    }
}

#[test]
fn obr_three_cycle_perturbation_and_phase_b_batch_reproducibility() {
    let inst = ft06_instance();
    let mut streams = paired_phase_b_context(3, "pert_test", "B17").all_streams();
    let (_, sched) =
        generate_matched_phase_b_start(&inst, InitializerId::A07, 16, &mut streams).expect("start");
    let mut eval = FastGraphEvaluator::new(&inst).expect("eval");
    eval.load_from_schedule(&sched).expect("load");
    let state = eval.analyze_current().expect("ok").expect("acyclic");
    let obr = eval.encode_canonical_obr(&state);
    let orig = obr.to_job_indices();

    let mut mutated_a = orig.clone();
    let mut rng_a = paired_phase_b_context(3, "pert_test", "B17")
        .all_streams()
        .perturbation;
    perturb_obr_three_cycle(&mut mutated_a, inst.num_jobs(), &mut rng_a).expect("3-cycle A");
    ObrChromosome::from_job_indices(&mutated_a)
        .validate(&inst)
        .expect("3-cycle must preserve OBR multiplicity");

    let diff_positions: Vec<usize> = (0..orig.len())
        .filter(|&i| orig[i] != mutated_a[i])
        .collect();
    assert_eq!(
        diff_positions.len(),
        3,
        "3-cycle must alter exactly 3 positions"
    );
    let distinct_orig_jobs: HashSet<usize> = diff_positions.iter().map(|&i| orig[i]).collect();
    assert_eq!(
        distinct_orig_jobs.len(),
        3,
        "3-cycle jobs must be pairwise distinct"
    );

    let mut mutated_b = orig.clone();
    let mut rng_b = paired_phase_b_context(3, "pert_test", "B17")
        .all_streams()
        .perturbation;
    perturb_obr_three_cycle(&mut mutated_b, inst.num_jobs(), &mut rng_b).expect("3-cycle B");
    assert_eq!(mutated_a, mutated_b);

    // Batch (multi-threaded) vs single reproducibility across stochastic metaheuristics
    let test_cfgs = [
        PhaseBConfigId::B09,
        PhaseBConfigId::B13,
        PhaseBConfigId::B17,
        PhaseBConfigId::B21,
        PhaseBConfigId::B29,
    ];
    let tasks: Vec<PhaseBBatchTask> = test_cfgs
        .iter()
        .map(|&cfg| PhaseBBatchTask {
            config_id: cfg,
            seed_context: paired_phase_b_context(5, "batch_ft06", cfg.code()),
            evaluation_budget: 400,
        })
        .collect();

    let batch_res = run_phase_b_batch(&inst, InitializerId::A07, 32, &tasks, 2).expect("batch");
    assert_eq!(batch_res.len(), test_cfgs.len());

    for (task, b_res) in tasks.iter().zip(batch_res.iter()) {
        let mut s = task.seed_context.all_streams();
        let (_, s_sched) =
            generate_matched_phase_b_start(&inst, InitializerId::A07, 32, &mut s).expect("start");
        let single_res = run_phase_b_from_schedule(&inst, task.config_id, &s_sched, 400, &mut s)
            .expect("single");
        assert_eq!(b_res.best_makespan, single_res.best_makespan);
        assert_eq!(b_res.evals_used, single_res.evals_used);
        assert_eq!(b_res.best_milestones, single_res.best_milestones);
        assert_eq!(
            b_res.best_machine_sequences,
            single_res.best_machine_sequences
        );
        assert!((b_res.anytime_cmax_auc - single_res.anytime_cmax_auc).abs() < 1e-12);
    }
}
