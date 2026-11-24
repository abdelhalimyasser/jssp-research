//! Deterministic rule-based EWMA Adaptive Gate for `C06`, `C09`, and `C11` (`D06` §4.3).

use crate::error::JsspError;

pub const ADAPTIVE_GATE_P_INIT: f64 = 0.50;
pub const ADAPTIVE_GATE_P_MIN: f64 = 0.15;
pub const ADAPTIVE_GATE_P_MAX: f64 = 0.85;
pub const ADAPTIVE_GATE_LAMBDA: f64 = 0.30;
pub const ADAPTIVE_GATE_ETA: f64 = 0.05;
const EPSILON_YIELD: f64 = 1e-9;
const HOLD_THRESHOLD: f64 = 1e-12;

/// Complete deterministic state of the rule-based Dual-View Adaptive Gate (`D06` §4.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveGateState {
    pub p: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub lambda: f64,
    pub eta: f64,
    pub w_perf: f64,
    pub w_div: f64,
    pub ewma_signal: f64,
}

/// Diagnostic breakdown from a single generation's gate update step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveGateStep {
    pub delta_perf: f64,
    pub delta_div: f64,
    pub raw_signal: f64,
    pub ewma_signal: f64,
    pub next_p: f64,
}

impl AdaptiveGateState {
    /// Initializes an adaptive gate controller (`C06`/`C11`: `w_perf=0.70, w_div=0.30`;
    /// `C09` diversity ablation: `w_perf=1.00, w_div=0.00`).
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] on invalid bounds or non-finite weights.
    pub fn new(w_perf: f64, w_div: f64) -> Result<Self, JsspError> {
        if !w_perf.is_finite()
            || !w_div.is_finite()
            || w_perf < 0.0
            || w_div < 0.0
            || (w_perf + w_div) <= 0.0
        {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!("invalid adaptive gate weights w_perf={w_perf}, w_div={w_div}"),
            });
        }
        Ok(Self {
            p: ADAPTIVE_GATE_P_INIT,
            p_min: ADAPTIVE_GATE_P_MIN,
            p_max: ADAPTIVE_GATE_P_MAX,
            lambda: ADAPTIVE_GATE_LAMBDA,
            eta: ADAPTIVE_GATE_ETA,
            w_perf,
            w_div,
            ewma_signal: 0.0,
        })
    }

    /// Computes bounded `Delta_perf in [-1, 1]` from normalized score means `S^L, S^B`
    /// and local-search improvement yields `(delta_sum, eval_sum)` (`D06` §4.3).
    #[must_use]
    pub fn compute_delta_perf(
        mean_s_lam: f64,
        mean_s_bald: f64,
        lam_delta_sum: u64,
        lam_ls_evals: usize,
        bald_delta_sum: u64,
        bald_ls_evals: usize,
    ) -> f64 {
        let yield_lam = (lam_delta_sum as f64) / (lam_ls_evals.max(1) as f64);
        let yield_bald = (bald_delta_sum as f64) / (bald_ls_evals.max(1) as f64);
        let yield_term = (yield_lam - yield_bald) / (yield_lam + yield_bald + EPSILON_YIELD);
        let quality_term = mean_s_lam - mean_s_bald;
        (0.5 * quality_term + 0.5 * yield_term).clamp(-1.0, 1.0)
    }

    /// Updates `ewma_signal` and `p` deterministically using generation `t` signals (`D06` §4.3).
    pub fn update(&mut self, delta_perf: f64, d_ham_lam: f64, d_ham_bald: f64) -> AdaptiveGateStep {
        let dp = delta_perf.clamp(-1.0, 1.0);
        let dd = (d_ham_lam - d_ham_bald).clamp(-1.0, 1.0);
        let g = self.w_perf * dp + self.w_div * dd;
        self.ewma_signal = (1.0 - self.lambda) * self.ewma_signal + self.lambda * g;
        if self.ewma_signal.abs() >= HOLD_THRESHOLD {
            self.p = (self.p + self.eta * self.ewma_signal).clamp(self.p_min, self.p_max);
        }
        AdaptiveGateStep {
            delta_perf: dp,
            delta_div: dd,
            raw_signal: g,
            ewma_signal: self.ewma_signal,
            next_p: self.p,
        }
    }
}
