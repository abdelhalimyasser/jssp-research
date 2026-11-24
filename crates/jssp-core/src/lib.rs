//! Core library crate for the deterministic classical JSSP research framework.
//!
//! Milestone M2 implements the classical rectangular JSSP domain model, Operation-Based
//! Representation (OBR) validation, deterministic Serial SGS earliest idle-gap insertion
//! decoder, independent schedule feasibility verification, and pure integer makespan evaluation.

pub mod decoder;
pub mod error;
pub mod feasibility;
pub mod model;
pub mod objective;
pub mod representation;
pub mod rng;

/// Canonical crate identifier for workspace and FFI verification.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Package version string resolved at compile time from `Cargo.toml`.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the `jssp-core` package version string.
#[must_use]
pub const fn crate_version() -> &'static str {
    CRATE_VERSION
}

/// Returns a fixed diagnostic identification string for the Rust core crate.
#[must_use]
pub const fn core_ident() -> &'static str {
    concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::{CRATE_NAME, CRATE_VERSION, core_ident, crate_version};

    #[test]
    fn test_core_metadata() {
        assert_eq!(CRATE_NAME, "jssp-core");
        assert_eq!(crate_version(), CRATE_VERSION);
        assert_eq!(core_ident(), "jssp-core/0.1.0");
    }
}
