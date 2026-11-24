//! Controlled Cross-View Migration exploratory ablation (`C11`, `D06` §5).
//!
//! Strictly excluded from base Dual-View claims. Every `T_mig = 10` generations, identifies the
//! stronger view `v*` by lowest best makespan (tie-break toward `L`) and copies the top
//! `k_mig = max(1, floor(0.10 * N_pairs))` unique genotypes from `v*` into the worst slots
//! of the complementary view, charging all re-evaluations to the unified `EvalCounter`.

use std::collections::HashSet;

use crate::model::JobId;
use crate::representation::ObrChromosome;

use super::individual::DualViewIndividual;

pub const MIGRATION_INTERVAL_GENS: usize = 10;

/// Selected source-view genotypes and destination pair slots for one `C11` migration event.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub migrate_into_baldwinian: bool,
    pub targets: Vec<(usize, ObrChromosome)>,
}

/// Constructs a deterministic migration plan copying up to `max(1, floor(0.10 * pop.len()))`
/// unique elite genotypes from the stronger view into the worst slots of the complementary view.
#[must_use]
pub fn plan_cross_view_migration(pop: &[DualViewIndividual]) -> MigrationPlan {
    if pop.is_empty() {
        return MigrationPlan {
            migrate_into_baldwinian: true,
            targets: Vec::new(),
        };
    }

    let best_lam = pop.iter().map(|p| p.lam_fitness).min().unwrap_or(u64::MAX);
    let best_bald = pop.iter().map(|p| p.bald_fitness).min().unwrap_or(u64::MAX);
    let source_is_lam = best_lam <= best_bald;

    let k_mig = ((pop.len() as f64) * 0.10).floor() as usize;
    let k_mig = k_mig.max(1).min(pop.len());

    let mut source_order: Vec<usize> = (0..pop.len()).collect();
    if source_is_lam {
        source_order.sort_by_key(|&i| (pop[i].lam_fitness, pop[i].id));
    } else {
        source_order.sort_by_key(|&i| (pop[i].bald_fitness, pop[i].id));
    }

    let mut seen: HashSet<&[JobId]> = HashSet::with_capacity(k_mig);
    let mut donors: Vec<ObrChromosome> = Vec::with_capacity(k_mig);
    for idx in source_order {
        let g = if source_is_lam {
            &pop[idx].lam_genotype
        } else {
            &pop[idx].bald_genotype
        };
        if seen.insert(g.genes()) {
            donors.push(g.clone());
            if donors.len() == k_mig {
                break;
            }
        }
    }

    // Rank destination slots descending by target view fitness (worst first; tie-break highest index).
    let mut dst_order: Vec<usize> = (0..pop.len()).collect();
    if source_is_lam {
        dst_order.sort_by(|&a, &b| {
            pop[b]
                .bald_fitness
                .cmp(&pop[a].bald_fitness)
                .then_with(|| b.cmp(&a))
        });
    } else {
        dst_order.sort_by(|&a, &b| {
            pop[b]
                .lam_fitness
                .cmp(&pop[a].lam_fitness)
                .then_with(|| b.cmp(&a))
        });
    }

    let targets = dst_order
        .into_iter()
        .zip(donors)
        .collect::<Vec<(usize, ObrChromosome)>>();

    MigrationPlan {
        migrate_into_baldwinian: source_is_lam,
        targets,
    }
}
