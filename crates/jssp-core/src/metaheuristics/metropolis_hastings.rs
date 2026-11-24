//! Note on pruned standalone Metropolis–Hastings (`B25`–`B28`, `D04-A`: `REJECTED`).
//!
//! Per Decision `D04-A` (`docs/algorithms/mh_vs_sa.md`), standalone Metropolis–Hastings (`B25`–`B28`)
//! is excluded from the primary Phase B matrix (reducing 67 -> 63 active configurations) because
//! omitting the reverse-proposal Hastings ratio duplicates fixed-temperature Simulated Annealing
//! (`B21`–`B24`), whereas exact Hastings correction incurs reverse-neighborhood enumeration and
//! rejects transitions with zero reverse proposal probability. Hybrid Metropolis-filtered ILS +
//! Steepest Descent (`B29`–`B32`, `D04-B`) is active in [`crate::metaheuristics::ils_mh`].

/// Returns the `D04-A` exclusion rationale when `"B25"`..=`"B28"` is referenced.
#[must_use]
pub fn pruned_standalone_mh_note(code: &str) -> Option<&'static str> {
    match code {
        "B25" | "B26" | "B27" | "B28" => Some(
            "pruned per approved Decision D04-A (redundant with constant-T SA; Phase B = 28 active configs)",
        ),
        _ => None,
    }
}
