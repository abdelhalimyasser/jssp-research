# Benchmark Ingestion, Parsing & Validation Data Flow (`M3-B`)

**Implementation:** [`src/jssp_research/benchmarks/`](file:///home/abdelhalimyasser/Downloads/jssp-research/src/jssp_research/benchmarks/)

---

## 1. End-to-End Ingestion Pipeline

```text
  Raw Benchmark File (disk / fixture)
        │
        ├──► sha256_file / sha256_bytes ──► raw_sha256 (verified if catalog/fixture pins it)
        │
        ▼
  Python Format Parser (PARSER_REGISTRY[format_id])
        │   • standard_jssp_0based_v1 (v1.0.0): JSPLib / OR-Library rectangular 0-based pairs
        │   • standard_jssp_1based_v1 (v1.0.0): Rectangular interleaved 1-based machine pairs
        │   • taillard_matrix_v1      (v1.0.0): Taillard (1993) dual Times + 1-based Machines matrices
        │
        ▼
  Mandatory Rust Validation Boundary: jssp_research.native.validate_instance(jobs)
        │   Enforces: J >= 1, M >= 1, rectangular J x M, each job route is a permutation
        │             of 0..M-1, strictly positive u64 durations (p > 0), no u64 sum overflow.
        │
        ▼
  ParsedJsspInstance + canonical_instance_sha256(...)
        │   Canonical JSON: {"jobs":[[[m,p],...],...],"num_jobs":J,"num_machines":M}
        │   Guarantees: identical canonical_sha256 across equivalent file encodings!
        │
        └──► Embedded into RunManifest.instance
```

---

## 2. Single Source of Truth Boundary

Python handles file I/O, format tokenization, line-numbered error diagnostics (`BenchmarkParseError`), SHA-256 checksumming, and catalog split tracking, **never** duplicating scheduling semantics. Every parsed instance must pass Rust `JsspInstance::new` validation before entering manifests or scheduling calls.

Supported vs. Deferred Models:
- **Supported (M3):** Rectangular classical non-preemptive JSSP (`standard_jssp_0based_v1`, `standard_jssp_1based_v1`, `taillard_matrix_v1`).
- **Explicitly Deferred (`D07-C`):** Variable-route / re-entrant Track 4 (`DCT_VARIABLE_ROUTE`) and industrial Track 5 (`BEL_INDUSTRIAL`) remain rejected by classical parsers and carry `format_id = None`, `status = DEFERRED_UNSUPPORTED_MODEL`.
