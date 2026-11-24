//! Static Fixed Gate policies for Dual-View configs `C05` (`p=0.50`), `C07` (`p=0.25`),
//! and `C08` (`p=0.75`) (`D06` §4.2).

use crate::error::JsspError;

/// Constant gate parameter weighting Lamarckian view (`p`) vs Baldwinian view (`1 - p`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedGate {
    pub p_lamarckian: f64,
}

impl FixedGate {
    /// Constructs and validates a fixed Model A gate weight `p in [0, 1]`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] if `p` is non-finite or outside `[0, 1]`.
    pub fn new(p_lamarckian: f64) -> Result<Self, JsspError> {
        if !p_lamarckian.is_finite() || !(0.0..=1.0).contains(&p_lamarckian) {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!("fixed gate p must lie in [0, 1], got {p_lamarckian}"),
            });
        }
        Ok(Self { p_lamarckian })
    }
}
