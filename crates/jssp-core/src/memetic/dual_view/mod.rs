//! Model A Dual-View Memetic Architecture (`C05`–`C09`, `C11`) and `C10` Unpaired Control (`D06`).

pub mod adaptive_gate;
pub mod fixed_gate;
pub mod individual;
pub mod ls_budget_investigation;
pub mod migration;
pub mod reproduction;

pub use adaptive_gate::{AdaptiveGateState, AdaptiveGateStep};
pub use fixed_gate::FixedGate;
pub use individual::{
    DualViewIndividual, NormalizedPairScores, normalize_dual_view_scores, ranked_dual_view_indices,
    select_dual_view_parent,
};
pub use ls_budget_investigation::{
    BranchLsTelemetry, DualViewExploratoryResult, DualViewExploratoryTask,
    DualViewLsAllocationConfig, ExploratoryLsStrategy, run_budget_limited_sa,
    run_dual_view_exploratory_batch, run_dual_view_ls_allocation,
};
pub use migration::{MIGRATION_INTERVAL_GENS, MigrationPlan, plan_cross_view_migration};
pub use reproduction::reproduce_dual_view_within_view;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::genetic::{PoxWorkspace, mutate_obr, ranked_population_indices, tournament_select};
use crate::initialization::generate_initial_population;
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::model::{JobId, JsspInstance};
use crate::representation::ObrChromosome;
use crate::rng::RunRngStreams;
use crate::telemetry::{EvalCounter, compute_obr_hamming_diversity};

use super::baldwinian::apply_baldwinian_preservation;
use super::lamarckian::apply_lamarckian_writeback;
use super::{PhaseCConfigId, PhaseCGaConfig, decode_and_local_search, verify_final_best};

/// Accumulated state and final telemetry returned by Dual-View (`C05`–`C09`, `C11`) or
/// Unpaired Dual-Pool (`C10`) execution before final result assembly.
pub(crate) struct DualExecutionSummary {
    pub generations_executed: usize,
    pub decode_evals: usize,
    pub ls_evals: usize,
    pub best_makespan: u64,
    pub best_found_eval: usize,
    pub best_found_generation: usize,
    pub feasible_final: bool,
    pub final_obr_diversity: f64,
    pub final_lam_diversity: f64,
    pub final_bald_diversity: f64,
    pub final_gate_p: Option<f64>,
    pub lam_best_makespan: u64,
    pub bald_best_makespan: u64,
    pub gate_p_trajectory: Vec<(usize, f64)>,
    pub best_milestones: Vec<(usize, u64)>,
    pub crossover_count: usize,
    pub mutation_count: usize,
    pub writeback_count: usize,
    pub migrations_applied: usize,
    pub best_chromosome: Vec<usize>,
    pub best_machine_sequences: Vec<Vec<(usize, usize)>>,
}

/// Evaluates both views of one child pair within the unified `counter` (charging both decodes
/// and bounded LS). Returns `Ok(None)` if global `counter` hits budget before decoding `L` (or returns
/// partial/raw fallback cleanly only when completing initialized pair).
#[allow(clippy::too_many_arguments)]
fn eval_dual_pair_views(
    pair_id: usize,
    lam_raw_chr: &ObrChromosome,
    bald_raw_chr: &ObrChromosome,
    is_init: bool,
    instance: &JsspInstance,
    config: &PhaseCGaConfig,
    counter: &mut EvalCounter,
    decoder_ws: &mut SerialDecoderWorkspace,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
    decode_evals: &mut usize,
    ls_evals: &mut usize,
    writeback_count: &mut usize,
    best_makespan: &mut u64,
    best_chr: &mut ObrChromosome,
    best_found_eval: &mut usize,
    best_found_gen: &mut usize,
    curr_gen: usize,
    milestones: &mut Vec<(usize, u64)>,
) -> Result<Option<DualViewIndividual>, JsspError> {
    let Some((mut lam_out, _)) = decode_and_local_search(
        instance,
        lam_raw_chr,
        config.local_search,
        is_init,
        counter,
        decoder_ws,
        ls_decoder_ws,
        graph_eval,
        streams,
    )?
    else {
        return Ok(None);
    };
    *decode_evals += 1;
    *ls_evals += lam_out.ls_evals_consumed;
    apply_lamarckian_writeback(instance, &mut lam_out)?;
    if lam_out.post_ls_makespan < lam_out.raw_makespan {
        *writeback_count += 1;
    }
    if lam_out.post_ls_makespan < *best_makespan {
        *best_makespan = lam_out.post_ls_makespan;
        *best_chr = lam_out.effective_genotype.clone();
        *best_found_eval = counter.total();
        *best_found_gen = curr_gen;
        milestones.push((*best_found_eval, *best_makespan));
    }

    let Some((mut bald_out, _)) = decode_and_local_search(
        instance,
        bald_raw_chr,
        config.local_search,
        is_init,
        counter,
        decoder_ws,
        ls_decoder_ws,
        graph_eval,
        streams,
    )?
    else {
        return Ok(None);
    };
    *decode_evals += 1;
    *ls_evals += bald_out.ls_evals_consumed;
    apply_baldwinian_preservation(instance, &mut bald_out)?;
    if bald_out.post_ls_makespan < *best_makespan {
        *best_makespan = bald_out.post_ls_makespan;
        *best_chr = bald_out.improved_genotype.clone();
        *best_found_eval = counter.total();
        *best_found_gen = curr_gen;
        milestones.push((*best_found_eval, *best_makespan));
    }

    Ok(Some(DualViewIndividual {
        id: pair_id,
        lam_genotype: lam_out.effective_genotype.clone(),
        lam_fitness: lam_out.post_ls_makespan,
        lam_ls_delta: lam_out.ls_delta(),
        lam_ls_evals: lam_out.ls_evals_consumed,
        bald_genotype: bald_out.effective_genotype.clone(),
        bald_fitness: bald_out.post_ls_makespan,
        bald_ls_delta: bald_out.ls_delta(),
        bald_ls_evals: bald_out.ls_evals_consumed,
    }))
}

/// Runs Model A Dual-View Memetic GA (`C05`–`C09`, `C11`) with `N_pairs` (`32` pairs = `64` genotypes).
#[allow(clippy::too_many_lines)]
pub(crate) fn run_dual_view_model_a(
    instance: &JsspInstance,
    config: &PhaseCGaConfig,
    counter: &mut EvalCounter,
    decoder_ws: &mut SerialDecoderWorkspace,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
) -> Result<DualExecutionSummary, JsspError> {
    let n_pairs = config.population_size / 2;
    let elite_pairs = (config.elitism_count / 2).max(1).min(n_pairs - 1);
    let num_jobs = instance.num_jobs();
    let chrom_len = instance.num_operations();

    enum GateController {
        Fixed(FixedGate),
        Adaptive(AdaptiveGateState),
    }
    impl GateController {
        fn p(&self) -> f64 {
            match self {
                Self::Fixed(f) => f.p_lamarckian,
                Self::Adaptive(a) => a.p,
            }
        }
    }

    let mut gate = match config.config_id {
        PhaseCConfigId::C05 => GateController::Fixed(FixedGate::new(0.50)?),
        PhaseCConfigId::C07 => GateController::Fixed(FixedGate::new(0.25)?),
        PhaseCConfigId::C08 => GateController::Fixed(FixedGate::new(0.75)?),
        PhaseCConfigId::C06 | PhaseCConfigId::C11 => {
            GateController::Adaptive(AdaptiveGateState::new(0.70, 0.30)?)
        }
        PhaseCConfigId::C09 => GateController::Adaptive(AdaptiveGateState::new(1.00, 0.00)?),
        other => {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!("{} is not a Model A Dual-View config", other.code()),
            });
        }
    };
    let use_migration = matches!(config.config_id, PhaseCConfigId::C11);

    let generated = generate_initial_population(config.initializer, instance, n_pairs, streams)?;
    let mut pop: Vec<DualViewIndividual> = Vec::with_capacity(n_pairs);
    let mut next_pair_id = 0usize;
    let mut decode_evals = 0usize;
    let mut ls_evals = 0usize;
    let mut writeback_count = 0usize;
    let mut crossover_count = 0usize;
    let mut mutation_count = 0usize;
    let mut migrations_applied = 0usize;

    let mut best_makespan = u64::MAX;
    let mut best_chr = generated.chromosomes[0].clone();
    let mut best_found_eval = 0usize;
    let mut best_found_gen = 0usize;
    let mut milestones: Vec<(usize, u64)> = Vec::new();
    let mut gate_p_trajectory: Vec<(usize, f64)> = vec![(0, gate.p())];

    for init_chr in &generated.chromosomes {
        let Some(ind) = eval_dual_pair_views(
            next_pair_id,
            init_chr,
            init_chr,
            true,
            instance,
            config,
            counter,
            decoder_ws,
            ls_decoder_ws,
            graph_eval,
            streams,
            &mut decode_evals,
            &mut ls_evals,
            &mut writeback_count,
            &mut best_makespan,
            &mut best_chr,
            &mut best_found_eval,
            &mut best_found_gen,
            0,
            &mut milestones,
        )?
        else {
            break;
        };
        next_pair_id += 1;
        pop.push(ind);
    }

    let mut pox_ws = PoxWorkspace::new(num_jobs, chrom_len);
    let mut buf1: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut buf2: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut generation = 0usize;

    // Initial adaptive update at generation 0
    if let GateController::Adaptive(ref mut adapt) = gate
        && pop.len() >= 2
    {
        let (scores, mean_sl, mean_sb) = normalize_dual_view_scores(&pop, adapt.p);
        let _ = &scores;
        let lam_ds: u64 = pop.iter().map(|i| i.lam_ls_delta).sum();
        let lam_ev: usize = pop.iter().map(|i| i.lam_ls_evals).sum();
        let bald_ds: u64 = pop.iter().map(|i| i.bald_ls_delta).sum();
        let bald_ev: usize = pop.iter().map(|i| i.bald_ls_evals).sum();
        let dp = AdaptiveGateState::compute_delta_perf(
            mean_sl, mean_sb, lam_ds, lam_ev, bald_ds, bald_ev,
        );
        let lam_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.lam_genotype.clone()).collect();
        let bald_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.bald_genotype.clone()).collect();
        let d_ham_l = compute_obr_hamming_diversity(&lam_chrs, num_jobs);
        let d_ham_b = compute_obr_hamming_diversity(&bald_chrs, num_jobs);
        adapt.update(dp, d_ham_l, d_ham_b);
        gate_p_trajectory[0] = (0, adapt.p);
    }

    while counter.remaining() > 0 && pop.len() >= 2 {
        let p_cur = gate.p();
        let (scores, _, _) = normalize_dual_view_scores(&pop, p_cur);
        let ranked = ranked_dual_view_indices(&pop, &scores);

        let mut next_pop: Vec<DualViewIndividual> = Vec::with_capacity(n_pairs);
        for &e_idx in ranked.iter().take(elite_pairs) {
            let mut elite = pop[e_idx].clone();
            elite.lam_ls_delta = 0;
            elite.lam_ls_evals = 0;
            elite.bald_ls_delta = 0;
            elite.bald_ls_evals = 0;
            next_pop.push(elite);
        }

        while next_pop.len() < n_pairs && counter.remaining() > 0 {
            let pa_idx = select_dual_view_parent(
                &pop,
                &scores,
                config.tournament_size,
                &mut streams.selection,
            )?;
            let pb_idx = select_dual_view_parent(
                &pop,
                &scores,
                config.tournament_size,
                &mut streams.selection,
            )?;
            let ((lam_c1, bald_c1), (lam_c2, bald_c2), cx, mt) = reproduce_dual_view_within_view(
                &pop[pa_idx],
                &pop[pb_idx],
                num_jobs,
                config.crossover_rate,
                config.mutation_rate,
                &mut pox_ws,
                (&mut buf1, &mut buf2),
                streams,
            )?;
            crossover_count += cx;
            mutation_count += mt;

            let gen_next = generation + 1;
            let Some(child1) = eval_dual_pair_views(
                next_pair_id,
                &lam_c1,
                &bald_c1,
                false,
                instance,
                config,
                counter,
                decoder_ws,
                ls_decoder_ws,
                graph_eval,
                streams,
                &mut decode_evals,
                &mut ls_evals,
                &mut writeback_count,
                &mut best_makespan,
                &mut best_chr,
                &mut best_found_eval,
                &mut best_found_gen,
                gen_next,
                &mut milestones,
            )?
            else {
                break;
            };
            next_pair_id += 1;
            next_pop.push(child1);

            if next_pop.len() < n_pairs && counter.remaining() > 0 {
                let Some(child2) = eval_dual_pair_views(
                    next_pair_id,
                    &lam_c2,
                    &bald_c2,
                    false,
                    instance,
                    config,
                    counter,
                    decoder_ws,
                    ls_decoder_ws,
                    graph_eval,
                    streams,
                    &mut decode_evals,
                    &mut ls_evals,
                    &mut writeback_count,
                    &mut best_makespan,
                    &mut best_chr,
                    &mut best_found_eval,
                    &mut best_found_gen,
                    gen_next,
                    &mut milestones,
                )?
                else {
                    break;
                };
                next_pair_id += 1;
                next_pop.push(child2);
            }
        }

        if next_pop.len() < 2 {
            break;
        }
        generation += 1;
        pop = next_pop;

        // Exploratory cross-view migration (C11 only) every T_mig = 10 generations
        if use_migration
            && generation.is_multiple_of(MIGRATION_INTERVAL_GENS)
            && counter.remaining() > 0
        {
            let plan = plan_cross_view_migration(&pop);
            for (slot, donor_chr) in plan.targets {
                if counter.remaining() == 0 {
                    break;
                }
                let Some((mut out, _)) = decode_and_local_search(
                    instance,
                    &donor_chr,
                    config.local_search,
                    false,
                    counter,
                    decoder_ws,
                    ls_decoder_ws,
                    graph_eval,
                    streams,
                )?
                else {
                    break;
                };
                decode_evals += 1;
                ls_evals += out.ls_evals_consumed;
                migrations_applied += 1;
                if plan.migrate_into_baldwinian {
                    apply_baldwinian_preservation(instance, &mut out)?;
                    pop[slot].bald_genotype = out.effective_genotype.clone();
                    pop[slot].bald_fitness = out.post_ls_makespan;
                } else {
                    apply_lamarckian_writeback(instance, &mut out)?;
                    pop[slot].lam_genotype = out.effective_genotype.clone();
                    pop[slot].lam_fitness = out.post_ls_makespan;
                }
                if out.post_ls_makespan < best_makespan {
                    best_makespan = out.post_ls_makespan;
                    best_chr = if plan.migrate_into_baldwinian {
                        out.improved_genotype
                    } else {
                        out.effective_genotype
                    };
                    best_found_eval = counter.total();
                    best_found_gen = generation;
                    milestones.push((best_found_eval, best_makespan));
                }
            }
        }

        if let GateController::Adaptive(ref mut adapt) = gate {
            let (_, mean_sl, mean_sb) = normalize_dual_view_scores(&pop, adapt.p);
            let lam_ds: u64 = pop.iter().map(|i| i.lam_ls_delta).sum();
            let lam_ev: usize = pop.iter().map(|i| i.lam_ls_evals).sum();
            let bald_ds: u64 = pop.iter().map(|i| i.bald_ls_delta).sum();
            let bald_ev: usize = pop.iter().map(|i| i.bald_ls_evals).sum();
            let dp = AdaptiveGateState::compute_delta_perf(
                mean_sl, mean_sb, lam_ds, lam_ev, bald_ds, bald_ev,
            );
            let lam_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.lam_genotype.clone()).collect();
            let bald_chrs: Vec<ObrChromosome> =
                pop.iter().map(|i| i.bald_genotype.clone()).collect();
            let d_ham_l = compute_obr_hamming_diversity(&lam_chrs, num_jobs);
            let d_ham_b = compute_obr_hamming_diversity(&bald_chrs, num_jobs);
            adapt.update(dp, d_ham_l, d_ham_b);
            gate_p_trajectory.push((generation, adapt.p));
        } else {
            gate_p_trajectory.push((generation, gate.p()));
        }
    }

    let lam_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.lam_genotype.clone()).collect();
    let bald_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.bald_genotype.clone()).collect();
    let mut all_chrs = Vec::with_capacity(lam_chrs.len() + bald_chrs.len());
    all_chrs.extend(lam_chrs.iter().cloned());
    all_chrs.extend(bald_chrs.iter().cloned());

    let lam_best = pop
        .iter()
        .map(|i| i.lam_fitness)
        .min()
        .unwrap_or(best_makespan);
    let bald_best = pop
        .iter()
        .map(|i| i.bald_fitness)
        .min()
        .unwrap_or(best_makespan);

    let (ver_cmax, ver_genes, ver_seqs) = verify_final_best(instance, &best_chr, decoder_ws)?;
    if ver_cmax < best_makespan {
        best_makespan = ver_cmax;
    }

    Ok(DualExecutionSummary {
        generations_executed: generation,
        decode_evals,
        ls_evals,
        best_makespan,
        best_found_eval,
        best_found_generation: best_found_gen,
        feasible_final: true,
        final_obr_diversity: compute_obr_hamming_diversity(&all_chrs, num_jobs),
        final_lam_diversity: compute_obr_hamming_diversity(&lam_chrs, num_jobs),
        final_bald_diversity: compute_obr_hamming_diversity(&bald_chrs, num_jobs),
        final_gate_p: Some(gate.p()),
        lam_best_makespan: lam_best,
        bald_best_makespan: bald_best,
        gate_p_trajectory,
        best_milestones: milestones,
        crossover_count,
        mutation_count,
        writeback_count,
        migrations_applied,
        best_chromosome: ver_genes,
        best_machine_sequences: ver_seqs,
    })
}

/// Runs Unpaired Dual-Population Control (`C10`): two completely independent memetic pools
/// (`32` LGA + `32` BGA = `64` genotypes total, shared `EvalCounter`, NO shared identity or gate).
#[allow(clippy::too_many_lines)]
pub(crate) fn run_unpaired_dual_pool_control(
    instance: &JsspInstance,
    config: &PhaseCGaConfig,
    counter: &mut EvalCounter,
    decoder_ws: &mut SerialDecoderWorkspace,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
) -> Result<DualExecutionSummary, JsspError> {
    let pool_size = config.population_size / 2;
    let pool_elites = (config.elitism_count / 2).max(1).min(pool_size - 1);
    let num_jobs = instance.num_jobs();
    let chrom_len = instance.num_operations();

    let generated = generate_initial_population(
        config.initializer,
        instance,
        config.population_size,
        streams,
    )?;
    let (init_l, init_b) = generated.chromosomes.split_at(pool_size);

    let mut lam_chrs: Vec<ObrChromosome> = Vec::with_capacity(pool_size);
    let mut lam_fit: Vec<u64> = Vec::with_capacity(pool_size);
    let mut bald_chrs: Vec<ObrChromosome> = Vec::with_capacity(pool_size);
    let mut bald_fit: Vec<u64> = Vec::with_capacity(pool_size);

    let mut decode_evals = 0usize;
    let mut ls_evals = 0usize;
    let mut writeback_count = 0usize;
    let mut crossover_count = 0usize;
    let mut mutation_count = 0usize;

    let mut best_makespan = u64::MAX;
    let mut best_chr = generated.chromosomes[0].clone();
    let mut best_found_eval = 0usize;
    let mut best_found_gen = 0usize;
    let mut milestones: Vec<(usize, u64)> = Vec::new();

    for c in init_l {
        let Some((mut out, _)) = decode_and_local_search(
            instance,
            c,
            config.local_search,
            true,
            counter,
            decoder_ws,
            ls_decoder_ws,
            graph_eval,
            streams,
        )?
        else {
            break;
        };
        decode_evals += 1;
        ls_evals += out.ls_evals_consumed;
        apply_lamarckian_writeback(instance, &mut out)?;
        if out.post_ls_makespan < out.raw_makespan {
            writeback_count += 1;
        }
        if out.post_ls_makespan < best_makespan {
            best_makespan = out.post_ls_makespan;
            best_chr = out.effective_genotype.clone();
            best_found_eval = counter.total();
        }
        lam_chrs.push(out.effective_genotype);
        lam_fit.push(out.post_ls_makespan);
    }

    for c in init_b {
        let Some((mut out, _)) = decode_and_local_search(
            instance,
            c,
            config.local_search,
            true,
            counter,
            decoder_ws,
            ls_decoder_ws,
            graph_eval,
            streams,
        )?
        else {
            break;
        };
        decode_evals += 1;
        ls_evals += out.ls_evals_consumed;
        apply_baldwinian_preservation(instance, &mut out)?;
        if out.post_ls_makespan < best_makespan {
            best_makespan = out.post_ls_makespan;
            best_chr = out.improved_genotype.clone();
            best_found_eval = counter.total();
        }
        bald_chrs.push(out.effective_genotype);
        bald_fit.push(out.post_ls_makespan);
    }
    if best_makespan < u64::MAX {
        milestones.push((best_found_eval, best_makespan));
    }

    let mut pox_ws = PoxWorkspace::new(num_jobs, chrom_len);
    let mut buf1: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut buf2: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut generation = 0usize;

    while counter.remaining() > 0 && lam_chrs.len() >= 2 && bald_chrs.len() >= 2 {
        // Pool L next generation (LGA)
        let lam_ranked = ranked_population_indices(&lam_fit);
        let mut next_l_chrs = Vec::with_capacity(pool_size);
        let mut next_l_fit = Vec::with_capacity(pool_size);
        for &e in lam_ranked.iter().take(pool_elites) {
            next_l_chrs.push(lam_chrs[e].clone());
            next_l_fit.push(lam_fit[e]);
        }

        // Pool B next generation (BGA)
        let bald_ranked = ranked_population_indices(&bald_fit);
        let mut next_b_chrs = Vec::with_capacity(pool_size);
        let mut next_b_fit = Vec::with_capacity(pool_size);
        for &e in bald_ranked.iter().take(pool_elites) {
            next_b_chrs.push(bald_chrs[e].clone());
            next_b_fit.push(bald_fit[e]);
        }

        let gen_next = generation + 1;
        while (next_l_chrs.len() < pool_size || next_b_chrs.len() < pool_size)
            && counter.remaining() > 0
        {
            if next_l_chrs.len() < pool_size && counter.remaining() > 0 {
                let pa =
                    tournament_select(&lam_fit, config.tournament_size, &mut streams.selection)?;
                let pb =
                    tournament_select(&lam_fit, config.tournament_size, &mut streams.selection)?;
                if pox_ws.crossover_into(
                    (lam_chrs[pa].genes(), lam_chrs[pb].genes()),
                    num_jobs,
                    config.crossover_rate,
                    &mut streams.crossover,
                    (&mut buf1, &mut buf2),
                )? {
                    crossover_count += 1;
                }
                for raw_buf in [&buf1, &buf2] {
                    if next_l_chrs.len() >= pool_size || counter.remaining() == 0 {
                        break;
                    }
                    let mut c = ObrChromosome::new(raw_buf.clone());
                    if mutate_obr(&mut c, config.mutation_rate, &mut streams.mutation)? {
                        mutation_count += 1;
                    }
                    let Some((mut out, _)) = decode_and_local_search(
                        instance,
                        &c,
                        config.local_search,
                        false,
                        counter,
                        decoder_ws,
                        ls_decoder_ws,
                        graph_eval,
                        streams,
                    )?
                    else {
                        break;
                    };
                    decode_evals += 1;
                    ls_evals += out.ls_evals_consumed;
                    apply_lamarckian_writeback(instance, &mut out)?;
                    if out.post_ls_makespan < out.raw_makespan {
                        writeback_count += 1;
                    }
                    if out.post_ls_makespan < best_makespan {
                        best_makespan = out.post_ls_makespan;
                        best_chr = out.effective_genotype.clone();
                        best_found_eval = counter.total();
                        best_found_gen = gen_next;
                        milestones.push((best_found_eval, best_makespan));
                    }
                    next_l_chrs.push(out.effective_genotype);
                    next_l_fit.push(out.post_ls_makespan);
                }
            }

            if next_b_chrs.len() < pool_size && counter.remaining() > 0 {
                let pa =
                    tournament_select(&bald_fit, config.tournament_size, &mut streams.selection)?;
                let pb =
                    tournament_select(&bald_fit, config.tournament_size, &mut streams.selection)?;
                if pox_ws.crossover_into(
                    (bald_chrs[pa].genes(), bald_chrs[pb].genes()),
                    num_jobs,
                    config.crossover_rate,
                    &mut streams.crossover,
                    (&mut buf1, &mut buf2),
                )? {
                    crossover_count += 1;
                }
                for raw_buf in [&buf1, &buf2] {
                    if next_b_chrs.len() >= pool_size || counter.remaining() == 0 {
                        break;
                    }
                    let mut c = ObrChromosome::new(raw_buf.clone());
                    if mutate_obr(&mut c, config.mutation_rate, &mut streams.mutation)? {
                        mutation_count += 1;
                    }
                    let Some((mut out, _)) = decode_and_local_search(
                        instance,
                        &c,
                        config.local_search,
                        false,
                        counter,
                        decoder_ws,
                        ls_decoder_ws,
                        graph_eval,
                        streams,
                    )?
                    else {
                        break;
                    };
                    decode_evals += 1;
                    ls_evals += out.ls_evals_consumed;
                    apply_baldwinian_preservation(instance, &mut out)?;
                    if out.post_ls_makespan < best_makespan {
                        best_makespan = out.post_ls_makespan;
                        best_chr = out.improved_genotype.clone();
                        best_found_eval = counter.total();
                        best_found_gen = gen_next;
                        milestones.push((best_found_eval, best_makespan));
                    }
                    next_b_chrs.push(out.effective_genotype);
                    next_b_fit.push(out.post_ls_makespan);
                }
            }
        }

        if next_l_chrs.len() < 2 || next_b_chrs.len() < 2 {
            break;
        }
        generation += 1;
        lam_chrs = next_l_chrs;
        lam_fit = next_l_fit;
        bald_chrs = next_b_chrs;
        bald_fit = next_b_fit;
    }

    let mut all_chrs = Vec::with_capacity(lam_chrs.len() + bald_chrs.len());
    all_chrs.extend(lam_chrs.iter().cloned());
    all_chrs.extend(bald_chrs.iter().cloned());
    let lam_best = lam_fit.iter().copied().min().unwrap_or(best_makespan);
    let bald_best = bald_fit.iter().copied().min().unwrap_or(best_makespan);
    let (ver_cmax, ver_genes, ver_seqs) = verify_final_best(instance, &best_chr, decoder_ws)?;
    if ver_cmax < best_makespan {
        best_makespan = ver_cmax;
    }

    Ok(DualExecutionSummary {
        generations_executed: generation,
        decode_evals,
        ls_evals,
        best_makespan,
        best_found_eval,
        best_found_generation: best_found_gen,
        feasible_final: true,
        final_obr_diversity: compute_obr_hamming_diversity(&all_chrs, num_jobs),
        final_lam_diversity: compute_obr_hamming_diversity(&lam_chrs, num_jobs),
        final_bald_diversity: compute_obr_hamming_diversity(&bald_chrs, num_jobs),
        final_gate_p: None,
        lam_best_makespan: lam_best,
        bald_best_makespan: bald_best,
        gate_p_trajectory: Vec::new(),
        best_milestones: milestones,
        crossover_count,
        mutation_count,
        writeback_count,
        migrations_applied: 0,
        best_chromosome: ver_genes,
        best_machine_sequences: ver_seqs,
    })
}
