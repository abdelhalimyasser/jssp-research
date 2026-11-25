//! Verification suite for Phase C Memetic & Dual-View GAs (`C01`–`C11`, `D06`, `D07`).

use jssp_core::decoder::SerialDecoderWorkspace;
use jssp_core::feasibility::validate_schedule;
use jssp_core::genetic::PoxWorkspace;
use jssp_core::initialization::InitializerId;
use jssp_core::memetic::baldwinian::apply_baldwinian_preservation;
use jssp_core::memetic::dual_view::{
    AdaptiveGateState, DualViewIndividual, FixedGate, normalize_dual_view_scores,
    plan_cross_view_migration, reproduce_dual_view_within_view, select_dual_view_parent,
};
use jssp_core::memetic::lamarckian::{MemeticEvalOutcome, apply_lamarckian_writeback};
use jssp_core::memetic::partial_lamarckian::apply_partial_lamarckian_even_ranks;
use jssp_core::memetic::{
    DualViewLsAllocationConfig, ExploratoryLsStrategy, PhaseCBatchTask, PhaseCConfigId,
    PhaseCGaConfig, default_phase_c_budget, run_dual_view_exploratory_batch,
    run_dual_view_ls_allocation, run_phase_c_batch, run_phase_c_ga,
};
use jssp_core::metaheuristics::PhaseBStrategy;
use jssp_core::model::{JobId, JsspInstance};
use jssp_core::neighborhoods::NeighborhoodId;
use jssp_core::representation::ObrChromosome;
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

fn phase_c_context(rep: u32, inst_id: &str, cfg_id: &str) -> SeedDerivationContext {
    SeedDerivationContext::from_frozen_replicate(
        rep,
        inst_id,
        PairingPolicy::PairedInitialAndScramble {
            pairing_group: "phase_c_shared_init".to_owned(),
            configuration_id: cfg_id.to_owned(),
        },
    )
    .expect("valid frozen seed context")
}

#[test]
fn lamarckian_baldwinian_and_plga_even_rank_semantics() {
    let inst = ft06_instance();
    let mut ws = SerialDecoderWorkspace::new(&inst);
    let raw_genes: Vec<usize> = (0..6).flat_map(|_| 0..6).collect();
    let imp_genes: Vec<usize> = (0..6).flat_map(|j| std::iter::repeat_n(j, 6)).collect();
    let raw_chr = ObrChromosome::from_job_indices(&raw_genes);
    let imp_chr = ObrChromosome::from_job_indices(&imp_genes);
    let sched = ws.decode(&inst, &imp_chr, true).expect("decode");

    let mut out_lga = MemeticEvalOutcome {
        original_genotype: raw_chr.clone(),
        improved_genotype: imp_chr.clone(),
        effective_genotype: raw_chr.clone(),
        raw_makespan: 80,
        post_ls_makespan: 60,
        ls_evals_consumed: 10,
        best_schedule: sched.clone(),
    };
    apply_lamarckian_writeback(&inst, &mut out_lga).expect("lga writeback");
    assert_eq!(out_lga.effective_genotype, imp_chr);

    let mut out_bga = out_lga.clone();
    apply_baldwinian_preservation(&inst, &mut out_bga).expect("bga preserve");
    assert_eq!(out_bga.effective_genotype, raw_chr);

    // PLGA deterministic 50% even-rank writeback
    let mut chrs = vec![raw_chr.clone(); 4];
    let imps = vec![Some(imp_chr.clone()); 4];
    // Fitnesses -> sorted rank order by (fit, idx): [3 (fit=50), 1 (fit=60), 2 (fit=70), 0 (fit=80)]
    // Even ranks (0, 2) => indices 3 and 2 receive write-back; odd ranks (1, 3) => 1 and 0 stay raw!
    let fits = vec![80, 60, 70, 50];
    let wb = apply_partial_lamarckian_even_ranks(&inst, &mut chrs, &imps, &fits).expect("plga");
    assert_eq!(wb, 2);
    assert_eq!(chrs[3], imp_chr);
    assert_eq!(chrs[2], imp_chr);
    assert_eq!(chrs[1], raw_chr);
    assert_eq!(chrs[0], raw_chr);
}

#[test]
fn model_a_gates_lineage_separation_and_migration_invariants() {
    assert!(FixedGate::new(-0.1).is_err());
    assert!(FixedGate::new(1.1).is_err());
    assert_eq!(FixedGate::new(0.25).expect("ok").p_lamarckian, 0.25);

    let mut adapt = AdaptiveGateState::new(0.70, 0.30).expect("ok");
    assert_eq!(adapt.p, 0.50);
    // Exact zero signal leaves p unchanged within 1e-12 hold band
    let step_zero = adapt.update(0.0, 0.5, 0.5);
    assert_eq!(step_zero.next_p, 0.50);
    // Strongly favor Baldwinian diversity over many generations -> clamped smoothly at >= p_min = 0.15
    for _ in 0..200 {
        adapt.update(-1.0, 0.0, 1.0);
    }
    assert!((adapt.p - 0.15).abs() < 1e-12);
    // Strongly favor Lamarckian -> clamped smoothly at <= p_max = 0.85
    for _ in 0..400 {
        adapt.update(1.0, 1.0, 0.0);
    }
    assert!((adapt.p - 0.85).abs() < 1e-12);

    // Lineage separation proof: POX must NEVER transfer genes from G^B pool into G^L pool.
    let lam_only =
        ObrChromosome::from_job_indices(&(0..6).flat_map(|_| 0..6).collect::<Vec<usize>>());
    let bald_only = ObrChromosome::from_job_indices(
        &(0..6)
            .flat_map(|j| std::iter::repeat_n(j, 6))
            .collect::<Vec<usize>>(),
    );
    let ind_a = DualViewIndividual {
        id: 1,
        lam_genotype: lam_only.clone(),
        lam_fitness: 70,
        lam_ls_delta: 0,
        lam_ls_evals: 0,
        bald_genotype: bald_only.clone(),
        bald_fitness: 60,
        bald_ls_delta: 0,
        bald_ls_evals: 0,
    };
    let ind_b = DualViewIndividual {
        id: 2,
        lam_genotype: lam_only.clone(),
        lam_fitness: 65,
        lam_ls_delta: 0,
        lam_ls_evals: 0,
        bald_genotype: bald_only.clone(),
        bald_fitness: 62,
        bald_ls_delta: 0,
        bald_ls_evals: 0,
    };

    let mut pox_ws = PoxWorkspace::new(6, 36);
    let mut b1: Vec<JobId> = Vec::new();
    let mut b2: Vec<JobId> = Vec::new();
    let mut streams = phase_c_context(0, "lineage_test", "C06").all_streams();
    let ((lc1, bc1), (lc2, bc2), _, _) = reproduce_dual_view_within_view(
        &ind_a,
        &ind_b,
        6,
        1.0,
        0.0, // zero mutation isolates POX strictly within view
        &mut pox_ws,
        (&mut b1, &mut b2),
        &mut streams,
    )
    .expect("reproduce");
    assert_eq!(lc1, lam_only);
    assert_eq!(lc2, lam_only);
    assert_eq!(bc1, bald_only);
    assert_eq!(bc2, bald_only);

    // Commensurable Model A normalization & pair-level selection
    let pop = vec![ind_a, ind_b];
    let (scores, _, _) = normalize_dual_view_scores(&pop, 0.5);
    assert!(scores[0].s_lam < scores[1].s_lam);
    assert!(scores[0].s_bald > scores[1].s_bald);
    let _picked = select_dual_view_parent(&pop, &scores, 2, &mut streams.selection).expect("pick");

    // Cross-view migration selects from stronger view into worst slots of weaker view
    let mut pop10 = Vec::new();
    for i in 0..10 {
        let g = ObrChromosome::from_job_indices(
            &(0..6)
                .flat_map(|_| (i..i + 6).map(|x| x % 6))
                .collect::<Vec<usize>>(),
        );
        pop10.push(DualViewIndividual {
            id: i,
            lam_genotype: g.clone(),
            lam_fitness: 80 + (i as u64),
            lam_ls_delta: 0,
            lam_ls_evals: 0,
            bald_genotype: g,
            bald_fitness: 60 + (i as u64),
            bald_ls_delta: 0,
            bald_ls_evals: 0,
        });
    }
    let mig = plan_cross_view_migration(&pop10);
    assert!(!mig.migrate_into_baldwinian); // bald (60) < lam (80)
    assert_eq!(mig.targets.len(), 1);
    assert_eq!(mig.targets[0].0, 9); // worst lam slot
}

#[test]
fn all_11_phase_c_configs_execute_within_equal_budget_and_verify_feasibility() {
    assert_eq!(PhaseCConfigId::ALL.len(), 11);
    let inst = ft06_instance();
    assert_eq!(default_phase_c_budget(6, 6), 6400);

    let budget = 900;
    let tasks: Vec<PhaseCBatchTask> = PhaseCConfigId::ALL
        .iter()
        .map(|&cfg| PhaseCBatchTask {
            config: PhaseCGaConfig::standard(
                cfg,
                InitializerId::A07,
                budget,
                NeighborhoodId::N7,
                PhaseBStrategy::SteepestDescent,
                50,
            ),
            seed_context: phase_c_context(1, "ft06", cfg.code()),
        })
        .collect();

    let batch_res = run_phase_c_batch(&inst, &tasks, 2).expect("phase C batch");
    assert_eq!(batch_res.len(), 11);

    let mut ws = SerialDecoderWorkspace::new(&inst);
    for (task, res) in tasks.iter().zip(batch_res.iter()) {
        assert_eq!(res.config_id, task.config.config_id.code());
        assert_eq!(res.genotype_capacity, 64);
        assert!(res.feasible_final);
        assert!(res.evals_used <= budget);
        assert_eq!(res.init_evals + res.search_evals, res.evals_used);
        assert_eq!(res.decode_evals + res.ls_evals, res.evals_used);
        assert!(res.best_makespan >= 55);

        // Verify returned best_chromosome via M2 Serial SGS + feasibility
        let best_obr = ObrChromosome::from_job_indices(&res.best_chromosome);
        let sched = ws.decode(&inst, &best_obr, true).expect("decode best");
        validate_schedule(&inst, &sched).expect("valid schedule");
        assert_eq!(sched.makespan(), res.best_makespan);

        // Check gate trajectory & migration properties per config
        match task.config.config_id {
            PhaseCConfigId::C05 => assert_eq!(res.final_gate_p, Some(0.50)),
            PhaseCConfigId::C07 => assert_eq!(res.final_gate_p, Some(0.25)),
            PhaseCConfigId::C08 => assert_eq!(res.final_gate_p, Some(0.75)),
            PhaseCConfigId::C06 | PhaseCConfigId::C09 | PhaseCConfigId::C11 => {
                let p = res.final_gate_p.expect("adaptive p must be recorded");
                assert!((0.15..=0.85).contains(&p));
                assert!(!res.gate_p_trajectory.is_empty());
            }
            PhaseCConfigId::C10 => {
                assert!(res.final_gate_p.is_none());
                assert!(res.final_lam_diversity.is_some());
                assert!(res.final_bald_diversity.is_some());
                assert_eq!(res.migrations_applied, 0);
            }
            _ => {
                assert!(res.final_gate_p.is_none());
                assert_eq!(res.migrations_applied, 0);
            }
        }
        if !matches!(task.config.config_id, PhaseCConfigId::C11) {
            assert_eq!(res.migrations_applied, 0);
        }

        // Verify single-thread parity with batch output
        let mut s = task.seed_context.all_streams();
        let single = run_phase_c_ga(&inst, &task.config, &mut s).expect("single");
        assert_eq!(single.best_makespan, res.best_makespan);
        assert_eq!(single.evals_used, res.evals_used);
        assert_eq!(single.best_chromosome, res.best_chromosome);
        assert_eq!(single.final_gate_p, res.final_gate_p);
    }
}

#[test]
fn exploratory_dv_ls_budget_parity_conservation_selection_and_short_sa() {
    let inst = ft06_instance();
    let budget = 1000;

    // 1. Control config (b_LS=50, f_LS=1.0, SteepestDescent, N5) must match C06 bit-for-bit.
    let c06_cfg = PhaseCGaConfig::standard(
        PhaseCConfigId::C06,
        InitializerId::A11,
        budget,
        NeighborhoodId::N5,
        PhaseBStrategy::SteepestDescent,
        50,
    );
    let mut s_c06 = phase_c_context(0, "ft06", "C06").all_streams();
    let res_c06 = run_phase_c_ga(&inst, &c06_cfg, &mut s_c06).expect("c06");

    let ctrl_cfg = DualViewLsAllocationConfig::standard(
        "DV_A4_B50".to_owned(),
        PhaseCConfigId::C06,
        InitializerId::A11,
        budget,
        NeighborhoodId::N5,
        ExploratoryLsStrategy::SteepestDescent,
        50,
        1.0,
    );
    let mut s_ctrl = phase_c_context(0, "ft06", "C06").all_streams();
    let res_ctrl = run_dual_view_ls_allocation(&inst, &ctrl_cfg, &mut s_ctrl).expect("dv_ctrl");

    assert_eq!(res_ctrl.best_makespan, res_c06.best_makespan);
    assert_eq!(res_ctrl.evals_used, res_c06.evals_used);
    assert_eq!(res_ctrl.generations_executed, res_c06.generations_executed);
    assert_eq!(res_ctrl.final_gate_p, res_c06.final_gate_p);
    assert_eq!(res_ctrl.best_chromosome, res_c06.best_chromosome);
    assert_eq!(res_ctrl.anytime_cmax_auc, res_c06.anytime_cmax_auc);

    // 2. Exact budget conservation across LS limits, candidate fractions, and strategies.
    let cases = [
        (
            "DV_A1_B10",
            ExploratoryLsStrategy::SteepestDescent,
            10usize,
            1.0_f64,
        ),
        (
            "DV_A2_B20",
            ExploratoryLsStrategy::SteepestDescent,
            20usize,
            1.0_f64,
        ),
        (
            "DV_B1_F020",
            ExploratoryLsStrategy::SteepestDescent,
            20usize,
            0.20_f64,
        ),
        (
            "DV_B2_F040",
            ExploratoryLsStrategy::SteepestDescent,
            20usize,
            0.40_f64,
        ),
        (
            "DV_B3_F060",
            ExploratoryLsStrategy::SteepestDescent,
            20usize,
            0.60_f64,
        ),
        (
            "DV_C2_FI",
            ExploratoryLsStrategy::FirstImprovement,
            20usize,
            0.40_f64,
        ),
        (
            "DV_C3_SA",
            ExploratoryLsStrategy::BudgetLimitedSa,
            20usize,
            0.40_f64,
        ),
    ];

    let mut ws = SerialDecoderWorkspace::new(&inst);
    let mut tasks = Vec::new();
    for &(exp_id, strat, b_ls, f_ls) in &cases {
        let cfg = DualViewLsAllocationConfig::standard(
            exp_id.to_owned(),
            PhaseCConfigId::C06,
            InitializerId::A11,
            budget,
            NeighborhoodId::N5,
            strat,
            b_ls,
            f_ls,
        );
        let ctx = phase_c_context(2, "ft06", "C06");
        let mut s1 = ctx.all_streams();
        let r1 = run_dual_view_ls_allocation(&inst, &cfg, &mut s1).expect(exp_id);
        let mut s2 = ctx.all_streams();
        let r2 = run_dual_view_ls_allocation(&inst, &cfg, &mut s2).expect("repro");
        assert_eq!(r1.best_makespan, r2.best_makespan);
        assert_eq!(r1.best_chromosome, r2.best_chromosome);
        assert_eq!(r1.evals_used, r2.evals_used);

        assert!(r1.feasible_final);
        assert!(r1.evals_used <= budget);
        assert_eq!(r1.init_evals + r1.search_evals, r1.evals_used);
        let branch_total = r1.lam_branch.decode_evals
            + r1.lam_branch.ls_evals
            + r1.bald_branch.decode_evals
            + r1.bald_branch.ls_evals;
        assert_eq!(branch_total, r1.evals_used);
        assert!(
            r1.lam_branch.decode_evals >= r1.bald_branch.decode_evals
                && r1.lam_branch.decode_evals - r1.bald_branch.decode_evals <= 1
        );
        assert_eq!(r1.diversity_trajectory.len(), r1.generations_executed + 1);

        let best_obr = ObrChromosome::from_job_indices(&r1.best_chromosome);
        let sched = ws.decode(&inst, &best_obr, true).expect("decode best");
        validate_schedule(&inst, &sched).expect("valid M2 schedule");
        assert_eq!(sched.makespan(), r1.best_makespan);

        if strat == ExploratoryLsStrategy::BudgetLimitedSa {
            assert!(r1.lam_branch.ls_calls > 0);
            assert!(r1.lam_branch.evaluated_moves > 0);
            assert!(r1.lam_branch.accepted_moves > 0);
        }

        tasks.push(jssp_core::memetic::DualViewExploratoryTask {
            config: cfg,
            seed_context: ctx,
        });
    }

    // 3. Batch parity check and lower fraction yielding higher generation count than 100%.
    let batch = run_dual_view_exploratory_batch(&inst, &tasks, 2).expect("batch");
    assert_eq!(batch.len(), cases.len());
    // DV_B1_F020 (index 2) must run strictly more generations than DV_A2_B20 (index 1).
    assert!(batch[2].generations_executed > batch[1].generations_executed);
    // Exact ceil(0.20 * 32) = 7 initial + ceil(0.20 * 31) = 7 per gen -> ls_calls per branch divisible or smaller than decodes
    assert!(batch[2].lam_branch.ls_calls < batch[2].lam_branch.decode_evals);

    // 4. Matched single-view C02 control: C02_CTRL_SD50 parity + C02_RETUNED_SA budget/selection checks.
    let c02_ref_cfg = PhaseCGaConfig::standard(
        PhaseCConfigId::C02,
        InitializerId::A11,
        budget,
        NeighborhoodId::N5,
        PhaseBStrategy::SteepestDescent,
        50,
    );
    let mut s_c02_ref = phase_c_context(0, "ft06", "C02").all_streams();
    let res_c02_ref = run_phase_c_ga(&inst, &c02_ref_cfg, &mut s_c02_ref).expect("c02_ref");

    let c02_ctrl_cfg = DualViewLsAllocationConfig::standard(
        "C02_CTRL_SD50".to_owned(),
        PhaseCConfigId::C02,
        InitializerId::A11,
        budget,
        NeighborhoodId::N5,
        ExploratoryLsStrategy::SteepestDescent,
        50,
        1.0,
    );
    let mut s_c02_ctrl = phase_c_context(0, "ft06", "C02").all_streams();
    let res_c02_ctrl =
        run_dual_view_ls_allocation(&inst, &c02_ctrl_cfg, &mut s_c02_ctrl).expect("c02_ctrl");

    assert_eq!(res_c02_ctrl.best_makespan, res_c02_ref.best_makespan);
    assert_eq!(res_c02_ctrl.evals_used, res_c02_ref.evals_used);
    assert_eq!(res_c02_ctrl.init_evals, res_c02_ref.init_evals);
    assert_eq!(res_c02_ctrl.search_evals, res_c02_ref.search_evals);
    assert_eq!(
        res_c02_ctrl.lam_branch.decode_evals,
        res_c02_ref.decode_evals
    );
    assert_eq!(res_c02_ctrl.lam_branch.ls_evals, res_c02_ref.ls_evals);
    assert_eq!(
        res_c02_ctrl.generations_executed,
        res_c02_ref.generations_executed
    );
    assert_eq!(res_c02_ctrl.best_chromosome, res_c02_ref.best_chromosome);
    assert_eq!(res_c02_ctrl.anytime_cmax_auc, res_c02_ref.anytime_cmax_auc);
    assert_eq!(res_c02_ctrl.bald_branch.decode_evals, 0);
    assert_eq!(res_c02_ctrl.bald_branch.ls_evals, 0);
    assert!(res_c02_ctrl.final_gate_p.is_none());

    let c02_retuned_cfg = DualViewLsAllocationConfig::standard(
        "C02_RETUNED_SA".to_owned(),
        PhaseCConfigId::C02,
        InitializerId::A11,
        budget,
        NeighborhoodId::N5,
        ExploratoryLsStrategy::BudgetLimitedSa,
        10,
        0.60,
    );
    let ctx_retuned = phase_c_context(2, "ft06", "C02");
    let mut s_ret1 = ctx_retuned.all_streams();
    let res_ret1 =
        run_dual_view_ls_allocation(&inst, &c02_retuned_cfg, &mut s_ret1).expect("c02_ret1");
    let mut s_ret2 = ctx_retuned.all_streams();
    let res_ret2 =
        run_dual_view_ls_allocation(&inst, &c02_retuned_cfg, &mut s_ret2).expect("c02_ret2");
    assert_eq!(res_ret1.best_makespan, res_ret2.best_makespan);
    assert_eq!(res_ret1.evals_used, res_ret2.evals_used);
    assert_eq!(res_ret1.generations_executed, res_ret2.generations_executed);
    assert_eq!(res_ret1.best_chromosome, res_ret2.best_chromosome);
    assert_eq!(res_ret1.best_milestones, res_ret2.best_milestones);
    assert_eq!(res_ret1.diversity_trajectory, res_ret2.diversity_trajectory);
    assert_eq!(res_ret1.lam_branch, res_ret2.lam_branch);
    assert_eq!(res_ret1.anytime_cmax_auc, res_ret2.anytime_cmax_auc);
    assert!(res_ret1.feasible_final);
    assert!(res_ret1.evals_used <= budget);
    assert_eq!(
        res_ret1.init_evals + res_ret1.search_evals,
        res_ret1.evals_used
    );
    assert_eq!(
        res_ret1.lam_branch.decode_evals + res_ret1.lam_branch.ls_evals,
        res_ret1.evals_used
    );
    assert_eq!(res_ret1.bald_branch.decode_evals, 0);
    assert_eq!(res_ret1.bald_branch.ls_evals, 0);
    assert!(res_ret1.final_gate_p.is_none());
    assert!(res_ret1.lam_branch.ls_calls > 0);
    assert!(res_ret1.lam_branch.ls_calls < res_ret1.lam_branch.decode_evals);
    assert!(res_ret1.lam_branch.evaluated_moves > 0);
    assert_eq!(
        res_ret1.diversity_trajectory.len(),
        res_ret1.generations_executed + 1
    );
    let ret_obr = ObrChromosome::from_job_indices(&res_ret1.best_chromosome);
    let ret_sched = ws.decode(&inst, &ret_obr, true).expect("decode retuned");
    validate_schedule(&inst, &ret_sched).expect("valid retuned schedule");
    assert_eq!(ret_sched.makespan(), res_ret1.best_makespan);
}
