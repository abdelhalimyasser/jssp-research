//! Chromosome representations for classical JSSP.

pub mod adapter;
pub mod obr;

pub use adapter::{
    AdapterAuditReport, AdapterRoundTripBundle, CanonicalOperationKey, EncodedObrResult,
    RoundTripStatus, audit_obr_against_target_graph, encode_graph_to_obr,
    round_trip_machine_sequences, round_trip_schedule,
};
pub use obr::ObrChromosome;
