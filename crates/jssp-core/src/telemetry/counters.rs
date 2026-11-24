//! Unified objective-evaluation budget accounting (`D07`).

use crate::error::JsspError;

/// Strict objective-evaluation counter enforcing matched computational budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalCounter {
    budget: usize,
    init_evals: usize,
    search_evals: usize,
}

impl EvalCounter {
    /// Creates a new counter with total allowed objective evaluations `budget`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] if `budget == 0`.
    pub fn new(budget: usize) -> Result<Self, JsspError> {
        if budget == 0 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: "objective evaluation budget must be strictly positive".to_owned(),
            });
        }
        Ok(Self {
            budget,
            init_evals: 0,
            search_evals: 0,
        })
    }

    /// Configured total evaluation budget.
    #[must_use]
    pub const fn budget(&self) -> usize {
        self.budget
    }

    /// Total evaluations consumed so far across initialization and search.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.init_evals + self.search_evals
    }

    /// Evaluations consumed during initial population decoding.
    #[must_use]
    pub const fn init_evals(&self) -> usize {
        self.init_evals
    }

    /// Evaluations consumed during post-initialization search.
    #[must_use]
    pub const fn search_evals(&self) -> usize {
        self.search_evals
    }

    /// Remaining evaluations strictly allowed under `budget`.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.budget
            .saturating_sub(self.init_evals + self.search_evals)
    }

    /// Charges one initial-population evaluation if within budget.
    pub fn try_consume_init(&mut self) -> bool {
        if self.total() >= self.budget {
            return false;
        }
        self.init_evals += 1;
        true
    }

    /// Charges one post-initialization search evaluation if within budget.
    pub fn try_consume_search(&mut self) -> bool {
        if self.total() >= self.budget {
            return false;
        }
        self.search_evals += 1;
        true
    }
}
