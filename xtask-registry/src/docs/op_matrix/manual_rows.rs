//! The scan construct rows the matrix carries by hand.
//!
//! Scan constructs are a tier vocabulary, not operations, and no registry
//! declares them, so the generator preserves this section verbatim and appends
//! the `[[op]]` rows it derives. Nothing else in the matrix is hand-written: an
//! `[[op]]` row that named something the registry never registered published a
//! surface the tree did not have.

/// The manual header every rendered matrix opens with.
pub(super) const SCAN_CONSTRUCT_MATRIX: &str = r#"# Manual scan construct tier data owned by VX-621/VX-622. Generated `[[op]]`
# rows below remain generator-owned.
scan_construct_tier_values = [
  "supported",
  "rejected",
  "approximated",
  "accelerator-only",
  "verifier-required",
]

scan_construct_route_values = [
  "native",
  "unsupported",
  "prefilter",
  "verifier",
  "external-accelerator",
  "host-reference",
]

[[scan_construct]]
id = "regular_exact_core"
tier = "supported"
dialect_class = "regular"
constructs = ["literal", "concatenation", "alternation", "bounded_repeat"]
diagnostic_code = "VYRE_SCAN_OK_EXACT_CORE"
user_diagnostic = "Exact regular constructs are eligible for native CPU and accelerator routes when backend capability checks pass."
approximation_policy = "exact"
verifier_required = false
accelerator_only = false
backend_routes = { cpu_ref = "native", cuda = "native", wgpu = "native", metal = "native", hyperscan = "native", vectorscan = "native", rust_regex = "native", dpu = "unsupported", fpga = "unsupported" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth/mod.rs", "xtask-registry/src/release/conformance_matrix/mod.rs"]

[[scan_construct]]
id = "unsupported_backtracking_constructs"
tier = "rejected"
dialect_class = "pcre-compatible"
constructs = ["backreference", "conditional_reference", "recursion", "subroutine_call"]
diagnostic_code = "VYRE_SCAN_UNSUPPORTED_BACKTRACKING_CONSTRUCT"
user_diagnostic = "Backtracking-only constructs are rejected unless a future verifier route has exact bounded semantics for the requested dialect."
approximation_policy = "none"
verifier_required = false
accelerator_only = false
backend_routes = { cpu_ref = "unsupported", cuda = "unsupported", wgpu = "unsupported", metal = "unsupported", hyperscan = "unsupported", vectorscan = "unsupported", rust_regex = "unsupported", dpu = "unsupported", fpga = "unsupported" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth/mod.rs", "xtask-registry/src/release/conformance_matrix/mod.rs"]

[[scan_construct]]
id = "lookaround_prefilter_constructs"
tier = "approximated"
dialect_class = "pcre-compatible"
constructs = ["positive_lookahead", "negative_lookahead", "fixed_width_lookbehind", "negative_lookbehind"]
diagnostic_code = "VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER"
user_diagnostic = "Lookaround constructs can only enter an approximate prefilter route when final match offsets are proven by a verifier."
approximation_policy = "broader-prefilter-plus-verifier"
verifier_required = true
accelerator_only = false
backend_routes = { cpu_ref = "host-reference", cuda = "prefilter", wgpu = "prefilter", metal = "prefilter", hyperscan = "prefilter", vectorscan = "prefilter", rust_regex = "unsupported", dpu = "unsupported", fpga = "prefilter" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth/mod.rs", "xtask-registry/src/release/conformance_matrix/mod.rs"]

[[scan_construct]]
id = "hardware_rule_database_constructs"
tier = "accelerator-only"
dialect_class = "external-rule-database"
constructs = ["bluefield_rule_set", "rof2_rule_database", "fpga_rule_image", "rxp_job"]
diagnostic_code = "VYRE_SCAN_ACCELERATOR_RULE_DATABASE_REQUIRED"
user_diagnostic = "External hardware rule databases are accelerator-only artifacts and must name the compiled rule digest before dispatch."
approximation_policy = "hardware-rule-database"
verifier_required = false
accelerator_only = true
backend_routes = { cpu_ref = "unsupported", cuda = "unsupported", wgpu = "unsupported", metal = "unsupported", hyperscan = "unsupported", vectorscan = "unsupported", rust_regex = "unsupported", dpu = "external-accelerator", fpga = "external-accelerator" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth/mod.rs", "xtask-registry/src/release/conformance_matrix/mod.rs"]

[[scan_construct]]
id = "capture_extraction_constructs"
tier = "verifier-required"
dialect_class = "capture"
constructs = ["capture_group", "named_capture", "submatch_offsets", "repeated_capture"]
diagnostic_code = "VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER"
user_diagnostic = "Capture extraction routes must preserve submatch spans through verifier output even when the accelerator only reports whole-match offsets."
approximation_policy = "whole-match-accelerator-plus-capture-verifier"
verifier_required = true
accelerator_only = false
backend_routes = { cpu_ref = "native", cuda = "verifier", wgpu = "verifier", metal = "verifier", hyperscan = "verifier", vectorscan = "verifier", rust_regex = "native", dpu = "unsupported", fpga = "verifier" }
proof_gates = ["conform/vyre-conform/tests/op_matrix_truth/mod.rs", "xtask-registry/src/release/conformance_matrix/mod.rs"]

"#;
