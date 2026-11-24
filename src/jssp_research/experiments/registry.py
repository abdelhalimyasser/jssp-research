"""Algorithm, decoder, and protocol constants recorded inside run manifests."""

from __future__ import annotations

from enum import StrEnum

CANONICAL_OBJECTIVE_ID = "makespan"
CANONICAL_TIME_SEMANTICS = "u64_positive_integer"
CANONICAL_DECODER_ID = "serial_sgs_earliest_gap"
CANONICAL_DECODER_VERSION = "1.0.0"


class ExperimentPhase(StrEnum):
    """Authoritative research phase identifiers."""

    PILOT = "pilot"
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    PHASE_C = "phase_c"


class ProtocolStage(StrEnum):
    """Experiment governance stage controlling dirty-tree and confirmatory checks."""

    DEVELOPMENT = "development"
    PILOT = "pilot"
    SCREENING = "screening"
    CONFIRMATORY = "confirmatory"
    STRESS_TEST = "stress_test"
