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
    PhaseCBatchTask, PhaseCConfigId, PhaseCGaConfig, default_phase_c_budget, run_phase_c_batch,
    run_phase_c_ga,
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
