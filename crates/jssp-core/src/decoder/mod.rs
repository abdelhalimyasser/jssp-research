//! Deterministic schedule decoders.

pub mod serial_insertion;

pub use serial_insertion::{SerialInsertionDecoder, decode_obr, decode_obr_append_only};
