//! Authoritative metadata for every registered gate.
//!
//! The implementation registry owns executable code. This table owns the
//! authoritative facts the runner must validate before execution: stable name,
//! help text, owner package, area membership, authoritative subject class,
//! exact generated paths, prerequisites, and mutation-proof method.
//! Registry and descriptor agreement is checked in both directions. Named subsets
//! are derived from `areas`; no second list of gate names exists.

use std::collections::BTreeSet;

pub use crate::gate_proof_validation::{validate_all_descriptors, validate_proof_symbol};

use crate::gate::GateDescriptor;

const FROZEN_CONTRACT_ARTIFACTS: &[&str] = &[
    "docs/frozen-traits/AlgebraicLaw.txt",
    "docs/frozen-traits/EnforceGate.txt",
    "docs/frozen-traits/ExprVisitor.txt",
    "docs/frozen-traits/Lowerable.txt",
    "docs/frozen-traits/MutationClass.txt",
    "docs/frozen-traits/PassBoundaryClass.txt",
    "docs/frozen-traits/VyreBackend.txt",
];

const PUBLIC_API_ARTIFACTS: &[&str] = &[
    "docs/public-api/vyre-aot.txt",
    "docs/public-api/vyre-debug.txt",
    "docs/public-api/vyre-driver-cuda.txt",
    "docs/public-api/vyre-driver-metal.txt",
    "docs/public-api/vyre-driver-reference.txt",
    "docs/public-api/vyre-driver-spirv.txt",
    "docs/public-api/vyre-driver-wgpu.txt",
    "docs/public-api/vyre-driver.txt",
    "docs/public-api/vyre-emit-metal.txt",
    "docs/public-api/vyre-emit-naga.txt",
    "docs/public-api/vyre-emit-ptx.txt",
    "docs/public-api/vyre-emit-spirv.txt",
    "docs/public-api/vyre-foundation.txt",
    "docs/public-api/vyre-libs.txt",
    "docs/public-api/vyre-lints.txt",
    "docs/public-api/vyre-lower.txt",
    "docs/public-api/vyre-macros.txt",
    "docs/public-api/vyre-megakernel.txt",
    "docs/public-api/vyre-pass-engine.txt",
    "docs/public-api/vyre-primitives.txt",
    "docs/public-api/vyre-reference.txt",
    "docs/public-api/vyre-runtime.txt",
    "docs/public-api/vyre-safetensors.txt",
    "docs/public-api/vyre-spec.txt",
    "docs/public-api/vyre.txt",
];

const TESTING_GUIDE_ARTIFACTS: &[&str] = &[
    "docs/testing/structure-gate.md",
    "docs/testing/vyre-aot.md",
    "docs/testing/vyre-bench.md",
    "docs/testing/vyre-conform-spec.md",
    "docs/testing/vyre-conform.md",
    "docs/testing/vyre-debug.md",
    "docs/testing/vyre-driver-cuda.md",
    "docs/testing/vyre-driver-metal.md",
    "docs/testing/vyre-driver-reference.md",
    "docs/testing/vyre-driver-spirv.md",
    "docs/testing/vyre-driver-wgpu.md",
    "docs/testing/vyre-driver.md",
    "docs/testing/vyre-emit-metal.md",
    "docs/testing/vyre-emit-naga.md",
    "docs/testing/vyre-emit-ptx.md",
    "docs/testing/vyre-emit-spirv.md",
    "docs/testing/vyre-foundation.md",
    "docs/testing/vyre-libs.md",
    "docs/testing/vyre-lints.md",
    "docs/testing/vyre-lower.md",
    "docs/testing/vyre-macros.md",
    "docs/testing/vyre-megakernel.md",
    "docs/testing/vyre-pass-engine.md",
    "docs/testing/vyre-primitives.md",
    "docs/testing/vyre-reference.md",
    "docs/testing/vyre-registry-link.md",
    "docs/testing/vyre-runtime.md",
    "docs/testing/vyre-safetensors.md",
    "docs/testing/vyre-spec.md",
    "docs/testing/vyre-test-support.md",
    "docs/testing/vyre.md",
    "docs/testing/xtask-evidence.md",
    "docs/testing/xtask-registry.md",
    "docs/testing/xtask.md",
];

/// Every gate descriptor, sorted by gate name.
pub static GATE_METADATA: &[GateDescriptor] = &[
    GateDescriptor {
        name: "abstraction-gate",
        help: "Enforce registered building-block boundaries",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::abstraction_gate::tests::a_finding_under_every_nesting_variant_is_reported",
    },
    GateDescriptor {
        name: "architecture-contract",
        help: "Enforce architecture-contract contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::architecture_contract::tests::a_lane_entry_may_name_a_directory_or_a_directory_glob",
    },
    GateDescriptor {
        name: "backend-extension",
        help: "Enforce backend-extension contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::frozen_contract::tests::a_needle_counts_only_when_its_terminator_follows",
    },
    GateDescriptor {
        name: "backend-matrix",
        help: "Judge the CUDA-first, WGPU-fallback backend policy. Proves, on any host, that every \\        backend implementation file the policy names exists and carries its implementation \\        tokens with no unresolved marker left in it, and that no backend production source \\        states a hidden fallback. Proves, from the recorded probe, that CUDA acquires first, \\        that the WGPU fallback acquires, that the preferred dispatch backend is never the \\        reference one, and that the host met the release GPU floor. The probe is only as \\        current as the run that recorded it; --write re-probes this host and rewrites the \\        artifact.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &["release/evidence/backends/backend-matrix.json"],
        prerequisites: &[],
        proof: "xtask_evidence::release::backend_matrix::feature_marker_tests::no_feature_marker_names_a_test_file",
    },
    GateDescriptor {
        name: "bench-baselines",
        help: "Enforce bench-baselines contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::bench::tests::every_missing_field_and_every_missing_section_is_reported_at_once",
    },
    GateDescriptor {
        name: "bench-coverage",
        help: "Enforce bench-coverage contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered benchmark cases",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::bench::tests::every_measured_dimension_is_judged_against_the_registry",
    },
    GateDescriptor {
        name: "bench-crossback",
        help: "Cross-validate every speedup cell across the three measured backend comparison \\        matrices. Proves every measured cell records positive speedup, is strictly reproducible \\        from its source probe, cites no placeholder, and is traceable back to its host, commit, \\        GPU model, driver version, and measurement timestamp. Proves nothing about benchmark \\        methodology: this gate reads the produced matrices and their source artifacts, never \\        starts a benchmark run. Run with --write to re-derive the markdown summary table.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &["release/evidence/benchmarks/cross-backend-comparison.md"],
        prerequisites: &[],
        proof: "xtask_evidence::bench::bench_crossback::tests::a_measurement_missing_any_provenance_field_is_a_finding",
    },
    GateDescriptor {
        name: "bench-release",
        help: "Judge the recorded release benchmark evidence across CUDA, WGPU, and CPU reference baselines. \\        Proves all 42 required release benchmark cases exist, passed on the recorded release host, \\        and achieved measured throughput and latency at or above the release floor. Proves zero \\        failed cases, zero unexplained regressions from the baseline, and that the recorded run \\        reached the required sample count. It measures nothing; regenerate the owned artifacts \\        with release-benchmarks before running this comparison gate.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_evidence::bench::bench_release::tests::bench_release_rejects_case_backend_drift_under_cuda_axes",
    },
    GateDescriptor {
        name: "bench-smoke-runtime",
        help: "Enforce bench-smoke-runtime contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::bench::tests::the_smoke_budget_comes_from_the_manifest",
    },
    GateDescriptor {
        name: "catalog",
        help: "Hold docs/generated/catalog.toml to the live operation inventory; --write regenerates it",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &["docs/generated/catalog.toml"],
        prerequisites: &[],
        proof: "xtask_registry::docs::catalog::tests::catalog_renders_subsystems_and_extracts_subsystem_names",
    },
    GateDescriptor {
        name: "check-tier-deps",
        help: "Enforce check-tier-deps contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::check_tier_deps::dependency_kind_tests::production_upward_workspace_inherited_dependency_fails",
    },
    GateDescriptor {
        name: "ci-matrix",
        help: "Enforce ci-matrix contracts",
        package: "xtask",
        areas: &["ci-rules"],
        subject: "ci workflows",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::ci_contract::tests::only_a_matrix_axis_value_counts",
    },
    GateDescriptor {
        name: "ci-registry",
        help: "Enforce ci-registry contracts",
        package: "xtask",
        areas: &["ci-rules"],
        subject: "ci workflows",
        artifacts: &["xtask/ci-registry.toml"],
        prerequisites: &[],
        proof: "crate::gates::ci_registry::tests::a_row_and_a_gate_that_do_not_pair_up_both_fail",
    },
    GateDescriptor {
        name: "ci-required",
        help: "Enforce ci-required contracts",
        package: "xtask",
        areas: &["ci-rules"],
        subject: "ci workflows",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::ci_contract::tests::a_fan_in_job_that_ignores_its_dependency_results_is_reported",
    },
    GateDescriptor {
        name: "ci-shell",
        help: "Enforce ci-shell contracts",
        package: "xtask",
        areas: &["ci-rules"],
        subject: "ci workflows",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::ci_contract::tests::a_step_declares_no_shell_when_no_key_at_its_depth_does",
    },
    GateDescriptor {
        name: "ci-steps",
        help: "Enforce ci-steps contracts",
        package: "xtask",
        areas: &["ci-rules"],
        subject: "ci workflows",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::ci_steps::tests::a_matrix_expression_is_not_a_selector_the_tree_can_refuse",
    },
    GateDescriptor {
        name: "cli-docs",
        help: "Enforce cli-docs contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::docs::cli_docs::tests::every_row_defect_is_its_own_finding",
    },
    GateDescriptor {
        name: "compile",
        help: "Lower a registered operation or an IR file to a target representation, verify the output against the target compiler (ptxas for PTX, spirv-val for SPIR-V, metal for MSL), and report binary size and register count; --op-id ID selects one operation, --file PATH selects an IR file, --target NAME (cuda, metal, wgpu, spirv, ptx) selects the backend",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::compile::tests::linked_target_compiler_emits_authenticated_payload",
    },
    GateDescriptor {
        name: "conformance-matrix",
        help: "Regenerate release/evidence/conformance/conformance-matrix.json from the live conformance suite across every registered operation and report each line the committed artifact disagrees on. Proves every registered operation has a conformance test, that all four backends achieve identical numeric results within declared precision bounds, that ULP drift stays below the gate ceiling, that no unsupported op is marked passing, and that every unsupported reason matches an approved class. Proves nothing about device execution: runs against the local host.",
        package: "xtask-registry",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &["release/evidence/conformance/conformance-matrix.json"],
        prerequisites: &[],
        proof: "xtask_registry::release::conformance_matrix::case_classes::tests::release_backend_case_rows_block_non_supported_rows_without_unsupported_evidence",
    },
    GateDescriptor {
        name: "contract-in-source",
        help: "Enforce contract-in-source contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::doc_contract::tests::a_path_in_code_is_not_a_deferral",
    },
    GateDescriptor {
        name: "crate-ownership",
        help: "Enforce crate-ownership contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &["docs/CRATE_GRAPH.md", "docs/OWNERSHIP.md"],
        prerequisites: &[],
        proof: "crate::gates::crate_registry::tests::inherited_and_local_features_are_unioned",
    },
    GateDescriptor {
        name: "crate-pages",
        help: "Enforce crate-pages contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::crate_pages::tests::an_empty_exclusion_section_is_reported",
    },
    GateDescriptor {
        name: "crate-readmes",
        help: "Enforce crate-readmes contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[
            "conform/vyre-conform-spec/README.md",
            "conform/vyre-conform/README.md",
            "structure-gate/README.md",
            "vyre-aot/README.md",
            "vyre-bench/README.md",
            "vyre-debug/README.md",
            "vyre-driver-cuda/README.md",
            "vyre-driver-metal/README.md",
            "vyre-driver-reference/README.md",
            "vyre-driver-spirv/README.md",
            "vyre-driver-wgpu/README.md",
            "vyre-driver/README.md",
            "vyre-emit-metal/README.md",
            "vyre-emit-naga/README.md",
            "vyre-emit-ptx/README.md",
            "vyre-emit-spirv/README.md",
            "vyre-foundation/README.md",
            "vyre-libs/README.md",
            "vyre-lints/README.md",
            "vyre-lower/README.md",
            "vyre-macros/README.md",
            "vyre-megakernel/README.md",
            "vyre-pass-engine/README.md",
            "vyre-primitives/README.md",
            "vyre-reference/README.md",
            "vyre-registry-link/README.md",
            "vyre-runtime/README.md",
            "vyre-safetensors/README.md",
            "vyre-spec/README.md",
            "vyre-test-support/README.md",
            "vyre/README.md",
            "xtask-evidence/README.md",
            "xtask-registry/README.md",
            "xtask/README.md",
        ],
        prerequisites: &[],
        proof: "crate::gates::crate_readmes::tests::an_unbalanced_marker_pair_is_a_finding",
    },
    GateDescriptor {
        name: "cross-target",
        help: "Check that the workspace compiles cleanly for every non-host release target (aarch64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc) through the workspace build wrapper; reports the first error per target",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::cross_target::tests::a_failure_with_no_error_line_still_names_itself",
    },
    GateDescriptor {
        name: "cuda-parity",
        help: "Enforce cuda-parity contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::backend_parity::tests::a_support_module_is_not_a_test_target",
    },
    GateDescriptor {
        name: "dep-drift",
        help: "Enforce dep-drift contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::dep_drift::tests::dep_drift_detects_mismatched_dependency_versions_and_ignores_workspace_inheritance",
    },
    GateDescriptor {
        name: "doc-claims",
        help: "Enforce doc-claims contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::doc_contract::tests::only_a_published_document_path_is_a_deferral",
    },
    GateDescriptor {
        name: "docs-check",
        help: "Enforce docs-check contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &["docs/SUMMARY.md", "docs/INDEX.md"],
        prerequisites: &[],
        proof: "crate::docs::docs_check::tests::every_authority_and_lifecycle_drift_is_reported",
    },
    GateDescriptor {
        name: "docs-coupling",
        help: "Enforce docs-coupling contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::docs::docs_coupling::tests::an_unreachable_base_ref_is_a_finding_and_not_a_crash",
    },
    GateDescriptor {
        name: "docs-references",
        help: "Enforce docs-references contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::docs_references::tests::a_single_backtick_span_is_read_and_a_double_one_is_not",
    },
    GateDescriptor {
        name: "docs-register",
        help: "Enforce docs-register contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::docs::docs_register::tests::a_root_page_row_naming_no_page_is_reported",
    },
    GateDescriptor {
        name: "dup-scan",
        help: "Enforce dup-scan contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &["xtask/dup-baseline.toml"],
        prerequisites: &[],
        proof: "crate::gates::dup_scan::tests::the_report_names_the_file_a_copy_was_made_from",
    },
    GateDescriptor {
        name: "error-codes",
        help: "Hold docs/generated/driver-error-codes.toml and docs/generated/error-codes.toml to the live driver and validation inventories; --write regenerates them",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "driver and validation error codes",
        artifacts: &[
            "docs/generated/driver-error-codes.toml",
            "docs/generated/error-codes.toml",
        ],
        prerequisites: &[],
        proof: "xtask_registry::docs::error_codes::tests::each_catalog_uses_the_renderer_of_the_crate_that_owns_it",
    },
    GateDescriptor {
        name: "evidence-paths",
        help: "Enforce evidence-paths contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::evidence_paths::tests::a_citation_that_does_not_resolve_is_reported",
    },
    GateDescriptor {
        name: "example-capability",
        help: "Enforce example-capability contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::example_capability::tests::an_unknown_placeholder_is_named",
    },
    GateDescriptor {
        name: "feature-isolation",
        help: "Enforce feature-isolation contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "workspace manifests",
        artifacts: &["xtask/feature-isolation.toml"],
        prerequisites: &[],
        proof: "crate::gates::feature_isolation::tests::a_recorded_outcome_that_disagrees_with_the_build_is_reported_both_ways",
    },
    GateDescriptor {
        name: "feature-matrix",
        help: "Regenerate release/evidence/metadata/feature-matrix.json from every workspace manifest \\            and report each line the committed artifact disagrees on. Proves every feature table \\            parses, every feature member resolves to a local feature, an optional dependency or a \\            dependency feature, every package that declares features declares a default policy, the \\            three release packages exist with empty defaults, and that vyre, vyre-driver-cuda and \\            vyre-driver-wgpu declare their release features. Proves nothing about whether any \\            feature selection compiles: that is feature-isolation.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "workspace manifests",
        artifacts: &["release/evidence/metadata/feature-matrix.json"],
        prerequisites: &[],
        proof: "crate::release::feature_matrix::tests::unresolved_feature_members_and_release_policy_are_enforced",
    },
    GateDescriptor {
        name: "feature-msrv",
        help: "Enforce feature-msrv contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::feature_msrv::tests::the_finding_carries_the_compiler_error",
    },
    GateDescriptor {
        name: "file-size",
        help: "Enforce file-size contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::file_size::tests::measured_rows_grant_no_headroom",
    },
    GateDescriptor {
        name: "frozen-contracts",
        help: "Enforce frozen-contracts contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: FROZEN_CONTRACT_ARTIFACTS,
        prerequisites: &[],
        proof: "crate::gates::frozen_contract::tests::a_snapshot_no_contract_claims_is_reported",
    },
    GateDescriptor {
        name: "gate-canon",
        help: "Enforce gate-canon contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::gate_canon::tests::a_row_with_nonzero_findings_fails_to_load",
    },
    GateDescriptor {
        name: "gate1",
        help: "Enforce Gate 1: loops <= 4 AND nodes <= 200 OR composed_fraction >= 60%",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::gate1::tests::the_finding_names_the_inline_work_it_collected",
    },
    GateDescriptor {
        name: "gpu-loudness",
        help: "Enforce gpu-loudness contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::gpu_loudness::tests::a_loud_abort_covers_any_shape",
    },
    GateDescriptor {
        name: "heuristic-audit",
        help: "Enforce heuristic-audit contracts",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::heuristic_audit::tests::a_plain_comment_marker_is_a_finding_at_its_line",
    },
    GateDescriptor {
        name: "host-oracle-elimination",
        help: "Enforce zero production host-oracle / cpu_ref mathematical implementations in shipping crates",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::host_oracle_elimination::tests::mutation_oracle_detection_catches_production_cpu_ref_fn",
    },
    GateDescriptor {
        name: "hot-path-blocking-wait",
        help: "Enforce hot-path-blocking-wait contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::hot_path::tests::blocking_waits_on_dispatch_paths_are_detected",
    },
    GateDescriptor {
        name: "hot-path-inventory",
        help: "Enforce hot-path-inventory contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::inventory_walk::tests::a_walk_on_a_lookup_path_is_reported_with_its_statement",
    },
    GateDescriptor {
        name: "hot-path-nested-rows",
        help: "Enforce hot-path-nested-rows contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::dispatch_surface::tests::a_row_returning_trait_with_no_slot_form_is_reported",
    },
    GateDescriptor {
        name: "hot-path-owned-dispatch",
        help: "Enforce hot-path-owned-dispatch contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::dispatch_surface::tests::a_trait_that_requires_the_owned_form_reports_the_requirement_and_the_copy",
    },
    GateDescriptor {
        name: "hot-path-reserve",
        help: "Enforce hot-path-reserve contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::hot_path::tests::a_reserve_argument_is_read_across_the_whole_statement",
    },
    GateDescriptor {
        name: "hot-path-scan",
        help: "Enforce hot-path-scan contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::hot_path_scan::tests::an_allocation_that_builds_an_error_is_not_measured_but_is_counted",
    },
    GateDescriptor {
        name: "hot-path-unbounded-cache",
        help: "Enforce hot-path-unbounded-cache contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::hot_path::tests::unbounded_cache_constructors_are_flagged",
    },
    GateDescriptor {
        name: "hot-path-unbounded-read",
        help: "Enforce hot-path-unbounded-read contracts",
        package: "xtask",
        areas: &["hot-path"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::hot_path::tests::unbounded_read_to_end_without_take_is_flagged",
    },
    GateDescriptor {
        name: "hygiene-matrix",
        help: "Enforce hygiene-matrix contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[
            "release/evidence/hygiene/hygiene-matrix.json",
            "release/evidence/hygiene/implementation-intake.json",
            "release/evidence/hygiene/threshold-policy.json",
            "release/evidence/hygiene/no-stubs-scan.json",
            "release/evidence/hygiene/no-hidden-fallback-scan.json",
            "release/evidence/hygiene/resource-bound-scan.json",
            "release/evidence/hygiene/error-surface-scan.json",
            "release/evidence/hygiene/cargo-wrapper-scan.json",
        ],
        prerequisites: &[],
        proof: "crate::gates::hygiene_matrix::tests::a_cargo_command_is_a_finding_when_a_comment_tells_a_reader_to_run_it",
    },
    GateDescriptor {
        name: "internal-dep-versions",
        help: "Enforce internal-dep-versions contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::manifest_contract::tests::a_rename_names_the_package_it_points_at",
    },
    GateDescriptor {
        name: "invariant-paths",
        help: "Enforce invariant-paths contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::evidence_paths::tests::a_cited_path_that_is_gitignored_is_reported",
    },
    GateDescriptor {
        name: "launch-state",
        help: "Regenerate release/evidence/final/public-launch-state.json from the launch completion \
        marker and the four prepublish gate artifacts, and report each line the committed \
        artifact disagrees on. Proves the recorded launch state matches the marker on disk, and \
        that each prepublish gate left an artifact carrying no blockers. A launch whose external \
        actions are still pending is recorded and noted, not reported: this gate runs on every \
        tree, and the gate that requires a closed launch is `vyre-release-gate --launch-complete`, \
        which reads this artifact in launch-complete mode. Proves nothing about whether the \
        external actions were really performed: the marker is written by the launch script and \
        this gate reads it, it does not contact crates.io or the git remote.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &["release/evidence/final/public-launch-state.json"],
        prerequisites: &[],
        proof: "crate::release::launch_state::tests::a_pending_launch_is_noted_and_a_missing_prepublish_artifact_is_reported",
    },
    GateDescriptor {
        name: "layering",
        help: "Enforce layering contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::layering::tests::a_neutral_crate_that_reaches_a_backend_api_only_through_a_third_party_crate_is_found",
    },
    GateDescriptor {
        name: "lego-composability",
        help: "Enforce LegoCheck 8: no operation is an island",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::composability::tests::island_operations_are_detected_and_connected_ops_pass",
    },
    GateDescriptor {
        name: "lego-composition-chains",
        help: "Enforce LegoCheck 6: non-leaf ops must have >= 1 child Region",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::composition_chain::tests::non_leaf_operations_without_children_are_flagged",
    },
    GateDescriptor {
        name: "lego-composition-depth",
        help: "Enforce LegoCheck 2: depth-of-composition",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::depth_of_composition::tests::below_quarter_composed_tier3_operation_fails_depth_gate",
    },
    GateDescriptor {
        name: "lego-cross-dialect",
        help: "Enforce LegoCheck 4: cross-dialect reach-through",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::cross_dialect::tests::the_substrate_is_an_exempt_target_and_still_an_audited_source",
    },
    GateDescriptor {
        name: "lego-exemption-liveness",
        help: "Enforce LegoCheck 0: every exemption is live",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::exemptions::tests::declared_leaf_classification_is_exact",
    },
    GateDescriptor {
        name: "lego-name-stems",
        help: "Enforce LegoCheck 9: name-stem collisions",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::name_stem::tests::leaf_stem_drops_first_underscore_suffix",
    },
    GateDescriptor {
        name: "lego-no-reinvention",
        help: "Enforce LegoCheck 1: no private reimplementation of primitives",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::no_reinvention::tests::ir_duplicate_analysis_judges_exactly_the_operations_that_carry_a_program",
    },
    GateDescriptor {
        name: "lego-operand-shapes",
        help: "Enforce LegoCheck 10: operand-shape duplicate review",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::operand_shape::tests::a_pair_that_agrees_only_where_the_bucket_key_reaches_is_not_a_duplicate",
    },
    GateDescriptor {
        name: "lego-primitive-coverage",
        help: "Enforce LegoCheck 3: primitive adoption coverage",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::primitive_coverage::tests::unregistered_primitive_family_fails_admission",
    },
    GateDescriptor {
        name: "lego-quick",
        help: "Enforce lego-quick contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lego_quick::tests::a_declared_dialect_edge_is_not_a_finding",
    },
    GateDescriptor {
        name: "lego-semantic-organization",
        help: "Enforce LegoCheck 11: semantic organization and file roles",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::semantic_organization::tests::overlapping_file_roles_fails_role_closure",
    },
    GateDescriptor {
        name: "lego-trend",
        help: "Enforce LegoCheck 7: composition trend ratchet",
        package: "xtask-registry",
        areas: &["lego-audit"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::lego_audit::trend::tests::every_intended_collapse_is_also_a_declared_leaf",
    },
    GateDescriptor {
        name: "lint-expect-fix",
        help: "Enforce lint-expect-fix contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lint_hygiene::tests::a_production_expect_owes_a_fix_and_a_test_item_does_not",
    },
    GateDescriptor {
        name: "lint-one-policy",
        help: "Enforce lint-one-policy contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lint_hygiene::tests::a_member_outside_the_workspace_policy_is_reported_and_an_inheriting_one_is_not",
    },
    GateDescriptor {
        name: "lint-unsafe-budget",
        help: "Enforce lint-unsafe-budget contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lint_hygiene::tests::only_a_real_override_counts_against_the_budget",
    },
    GateDescriptor {
        name: "lint-unsafe-justification",
        help: "Enforce lint-unsafe-justification contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lint_hygiene::tests::a_quoted_unsafe_block_is_data_and_a_real_one_still_needs_its_justification",
    },
    GateDescriptor {
        name: "list-ops",
        help: "Hold docs/generated/op-inventory.toml to the live operation registry; --write regenerates it",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &["docs/generated/op-inventory.toml"],
        prerequisites: &[],
        proof: "xtask_registry::docs::list_ops::tests::list_ops_renders_canonical_toml_schema",
    },
    GateDescriptor {
        name: "lockfile-clean",
        help: "Enforce lockfile-clean contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::lockfile::tests::lockfile_findings_detect_porcelain_diff_and_pass_clean",
    },
    GateDescriptor {
        name: "metadata-matrix",
        help: "Regenerate release/evidence/metadata/metadata-matrix.json from every workspace \\        manifest and report each line the committed artifact disagrees on. Proves every package \\        declares its license, repository, description, authors, and publish status, that \\        workspace-inherited metadata is resolved, that the release version is uniform across the \\        workspace, and that no unreviewed package is marked for publication. Proves nothing about \\        what is published on crates.io: every fact here is read from this checkout.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "workspace manifests",
        artifacts: &["release/evidence/metadata/metadata-matrix.json"],
        prerequisites: &[],
        proof: "crate::release::metadata_matrix::tests::release_classification_is_vyre_owned",
    },
    GateDescriptor {
        name: "metal-parity",
        help: "Enforce metal-parity contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::metal_parity::tests::a_destination_that_opens_an_option_is_refused",
    },
    GateDescriptor {
        name: "neutral-crates",
        help: "Enforce neutral-crates contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::layering::tests::every_backend_word_is_reported_from_the_list_rather_than_a_sample",
    },
    GateDescriptor {
        name: "op-matrix",
        help: "Hold docs/optimization/OP_MATRIX.toml to the live operation schema; --write regenerates it",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &["docs/optimization/OP_MATRIX.toml"],
        prerequisites: &[],
        proof: "xtask_registry::docs::op_matrix::validation::tests::an_op_with_no_live_registration_blocks_the_matrix",
    },
    GateDescriptor {
        name: "op-names",
        help: "Enforce op-names contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::op_names::tests::rejects_every_banned_shape_and_accepts_a_canonical_name",
    },
    GateDescriptor {
        name: "operation-schema",
        help: "Hold docs/generated/OP_SCHEMA.json to the live operation registry; --write regenerates it",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &["docs/generated/OP_SCHEMA.json"],
        prerequisites: &[],
        proof: "xtask_registry::docs::operation_schema::tests::the_architecture_contract_pins_the_same_operation_schema_version",
    },
    GateDescriptor {
        name: "optimization-corpus",
        help: "Regenerate the five artifacts under release/evidence/optimization from the semantic \\        Program optimizer corpus and report each line the committed copies disagree on. Proves \\        the corpus reaches its case floor, that every case verifies after optimization, that at \\        least one pass instance changed a program, that no case failed to converge, that every \\        required family carries its minimum case count, that no case id repeats, and that no \\        optimizer pass id repeats. Proves nothing about runtime performance: the corpus is \\        optimized and re-verified in process, never executed on a device.",
        package: "xtask-registry",
        areas: &["prepublish", "release-evidence"],
        subject: "registered IR corpus",
        artifacts: &["release/evidence/optimization/optimization-corpus.json", "release/evidence/optimization/optimization-corpus-contracts.json", "release/evidence/optimization/optimization-family-manifest.json", "release/evidence/optimization/optimization-case-manifest.json", "release/evidence/optimization/optimizer-pass-manifest.json"],
        prerequisites: &[],
        proof: "xtask_registry::release::optimization_corpus::tests::corpus_completeness_and_hex_encoding_are_exact",
    },
    GateDescriptor {
        name: "optimization-docs",
        help: "Hold docs/generated/optimizer-passes.toml to the live pass registry and supplemental catalog; --write regenerates it",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered optimizer passes",
        artifacts: &["docs/generated/optimizer-passes.toml"],
        prerequisites: &[],
        proof: "xtask_registry::docs::optimization_docs::tests::a_dropped_supplemental_rule_row_is_reported_missing",
    },
    GateDescriptor {
        name: "optimization-matrix",
        help: "Regenerate release/evidence/optimization/optimization-integration-matrix.json from the \\        live optimizer pass catalog and report every line the committed artifact disagrees on. \\        Proves the artifact lists exactly the catalog entries the source registers, that no pass \\        id repeats, and that every entry names an owner, invariant, proof and benchmark. Proves \\        nothing about whether a pass is correct, ever fires, or improves anything: the named \\        proof and benchmark are strings the catalog carries, not results this gate reads.",
        package: "xtask-registry",
        areas: &["prepublish", "release-evidence"],
        subject: "registered IR corpus",
        artifacts: &["release/evidence/optimization/optimization-integration-matrix.json"],
        prerequisites: &[],
        proof: "xtask_registry::release::optimization_matrix::tests::optimization_matrix_detects_incomplete_catalog_entries",
    },
    GateDescriptor {
        name: "oracle-sweeps",
        help: "Enforce oracle-sweeps contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::oracle_sweeps::tests::a_sweep_source_names_the_member_directory_that_holds_it",
    },
    GateDescriptor {
        name: "package-readiness",
        help: "Enforce the release package readiness gate across every publishable workspace crate. \\        Proves all required packages exist, carry matching version declarations, contain only \\        allowed license tokens, and that archive contents match the published artifact \\        manifests byte-for-byte. Proves zero unpack errors, zero forbidden file extensions, \\        and that package sizes stay within declared bandwidth budgets.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "workspace manifests",
        artifacts: &["release/evidence/package/publish-readiness.json"],
        prerequisites: &[],
        proof: "crate::release::package_readiness::archive_contracts::malformed_package_content_evidence_reports_every_failed_invariant",
    },
    GateDescriptor {
        name: "parity-testing-isolated",
        help: "Enforce parity-testing-isolated contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::parity_testing::tests::allows_only_the_declaration_and_a_development_dependency",
    },
    GateDescriptor {
        name: "path-deps-resolve",
        help: "Enforce path-deps-resolve contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::manifest_contract::tests::path_resolution_is_lexical_and_bounded",
    },
    GateDescriptor {
        name: "placement-predicates",
        help: "Enforce placement-predicates contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::placement_predicate::tests::a_shell_left_by_a_departed_member_is_reported",
    },
    GateDescriptor {
        name: "platform-boundary",
        help: "Enforce platform-boundary contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::platform_boundary::tests::finds_consumer_names_in_comments_but_not_identifiers",
    },
    GateDescriptor {
        name: "platform-consumer-docs",
        help: "Enforce platform-consumer-docs contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::platform_docs::tests::the_exemption_covers_a_directory_and_a_bare_name",
    },
    GateDescriptor {
        name: "print-composition",
        help: "Walk the decomposition chain of every registered operation; --op-id ID narrows to one",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered IR corpus",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::print_composition::tests::empty_region_bodies_are_detected_as_findings",
    },
    GateDescriptor {
        name: "program-wire-fields",
        help: "Enforce program-wire-fields contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::frozen_contract::tests::only_the_program_struct_fields_are_read",
    },
    GateDescriptor {
        name: "proptest-coverage",
        help: "Enforce proptest-coverage contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered operations",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::proptest_coverage::tests::proptest_marker_detection_and_floor_accounting",
    },
    GateDescriptor {
        name: "public-api-paths",
        help: "Enforce public-api-paths contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &["xtask/public-api-paths.toml"],
        prerequisites: &[],
        proof: "crate::gates::public_api_paths::tests::a_type_reachable_through_two_modules_is_one_duplicate",
    },
    GateDescriptor {
        name: "public-api-snapshot",
        help: "Enforce public-api-snapshot contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: PUBLIC_API_ARTIFACTS,
        prerequisites: &[],
        proof: "crate::gates::public_api::tests::a_snapshot_for_an_unpublished_package_is_a_finding",
    },
    GateDescriptor {
        name: "readback-ring",
        help: "Enforce readback-ring contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::frozen_contract::tests::neither_a_default_body_nor_a_comment_is_part_of_the_contract",
    },
    GateDescriptor {
        name: "release-benchmarks",
        help: "Hold release benchmark evidence artifacts under release/evidence/benchmarks/ to the \
        live benchmark registry and the committed evidence measurements. Proves every measured \
        benchmark case in the release suite exists in the registry, that no case is duplicated, \
        that measured speedups are positive and reproducible, and that no required release metric \
        is missing. The evidence is read from disk; --write updates the recorded artifacts.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: crate::artifact_paths::RELEASE_BENCHMARKS_ARTIFACTS,
        prerequisites: &[],
        proof: "xtask_evidence::bench::release_benchmarks::run::tests::authoritative_descriptor_declares_exact_release_benchmarks_artifacts",
    },
    GateDescriptor {
        name: "release-conformance",
        help: "Enforce the release conformance suite across the selected CUDA, WGPU, and CPU \
        reference backends. Proves every operation required by the current release OP_MATRIX \
        exists exactly once per selected backend and executes against vyre-reference. Non-F32 \
        outputs are byte-exact and F32 outputs stay within the Program-derived ULP cap. Proves \
        zero missing release rows, duplicate operation ids, malformed summaries, failed pairs, \
        and blockers. Runs the compiled suite on this host; CUDA and WGPU selections require \
        hardware acceleration. Run with --write to refresh release conformance evidence artifacts.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &[
            "release/evidence/conformance/cuda-conformance.json",
            "release/evidence/conformance/reference-conformance.json",
            "release/evidence/conformance/release-gate-log.json",
            "release/evidence/conformance/wgpu-conformance.json",
        ],
        prerequisites: &[],
        proof: "crate::release::release_conformance::tests::diff_summary_validation_rejects_missing_and_wrong_backend_fields",
    },
    GateDescriptor {
        name: "release-docs",
        help: "Hold docs/releases/0.7.0.md to the current release facts. Proves the document exists, \\        states the release date, links every published crate at its released version, and \\        names every Category A operation added or renamed in this release train. Proves \\        nothing about historical releases: only the current release document is held to \\        this check.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "owned documentation pages",
        artifacts: &["CHANGELOG.md", "release/evidence/docs/release-notes-body.md"],
        prerequisites: &[],
        proof: "crate::release::release_docs::tests::a_changelog_with_no_released_section_is_refused",
    },
    GateDescriptor {
        name: "release-evidence",
        help: "Regenerate release/evidence/final/release-evidence-run.json and expected-artifacts.json \
        and report each line the committed copies disagree on. Proves every required generator \
        declares at least one expected artifact, and that every declared artifact exists, is \
        non-empty, is readable and carries provenance. Proves nothing about whether those \
        generators pass: it no longer runs them. Each one is a registered gate, so the sweep \
        runs it and fails on it directly rather than through a spawn this gate reports \
        second-hand.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &[
            "release/evidence/final/expected-artifacts.json",
            "release/evidence/final/release-evidence-run.json",
        ],
        prerequisites: &[],
        proof: "xtask_evidence::release::release_evidence::tests::artifact_status_rejects_public_boundary_leaks",
    },
    GateDescriptor {
        name: "release-workload-matrix",
        help: "Rebuild release/evidence/benchmarks/release-workload-matrix.json from the benchmark case \
        registry and the bench target manifest, and report each line the committed artifact \
        disagrees on. Proves every required release workload family matches at least one \
        registered case, that each family naming a CPU state-of-the-art baseline has one, and \
        that the matrix carries no blockers. Proves nothing about any measurement: no benchmark \
        runs here and no artifact any family names is read.",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &["release/evidence/benchmarks/release-workload-matrix.json"],
        prerequisites: &[],
        proof: "xtask_evidence::release::release_workload_matrix::tests::the_committed_matrix_body_is_what_the_registry_derives",
    },
    GateDescriptor {
        name: "repo-hygiene",
        help: "Enforce repo-hygiene contracts",
        package: "xtask",
        areas: &["prepublish"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::repo_hygiene::tests::a_double_extension_archive_is_still_an_artifact",
    },
    GateDescriptor {
        name: "script-ledger",
        help: "Enforce script-ledger contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &["xtask/script-assertion-ledger.md"],
        prerequisites: &[],
        proof: "crate::gates::script_ledger::tests::a_ledger_missing_a_structural_heading_is_an_error_not_a_clean_run",
    },
    GateDescriptor {
        name: "shader-source",
        help: "Enforce shader-source contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::shader_source::tests::only_shader_owning_paths_may_hold_shader_syntax",
    },
    GateDescriptor {
        name: "shrink",
        help: "Delta-debug every registered corpus case that fails its oracle down to a minimal reproducer; --program ID narrows to one, --oracle PATH replaces the oracle",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered IR corpus",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::shrink::tests::wire_round_trip_oracle_verifies_node_count_integrity",
    },
    GateDescriptor {
        name: "single-backlog",
        help: "Enforce single-backlog contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::repo_hygiene::tests::a_redirect_is_required_only_where_policy_is_tracked_and_a_policy_copy_is_always_reported",
    },
    GateDescriptor {
        name: "source-include-module",
        help: "Enforce source-include-module contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::source_reachability::include_module_tests::a_tracked_include_is_reported_and_a_generated_one_is_not",
    },
    GateDescriptor {
        name: "source-parses",
        help: "Enforce source-parses contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::source_reachability::tests::includes_and_trybuild_patterns_are_read",
    },
    GateDescriptor {
        name: "source-reachability",
        help: "Enforce source-reachability contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "tracked source files",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::source_reachability::tests::declarations_come_from_code_only",
    },
    GateDescriptor {
        name: "spirv-parity",
        help: "Enforce spirv-parity contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::backend_parity::tests::only_a_target_requiring_the_feature_counts_as_gated",
    },
    GateDescriptor {
        name: "test-material-placement",
        help: "Enforce test-material-placement contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::test_material::tests::only_a_crate_that_declares_an_edge_can_be_a_referrer",
    },
    GateDescriptor {
        name: "testing-guides",
        help: "Enforce testing-guides contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "registered system invariants",
        artifacts: TESTING_GUIDE_ARTIFACTS,
        prerequisites: &[],
        proof: "crate::gates::testing_guides::tests::a_field_missing_at_every_level_is_a_finding",
    },
    GateDescriptor {
        name: "trace-f32",
        help: "Run the recorded test inputs of every registered operation through the reference; --op-id ID narrows to one",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered IR corpus",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::trace_f32::tests::render_run_formats_hex_buffer_literals",
    },
    GateDescriptor {
        name: "unification",
        help: "Enforce unification contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::unification::tests::a_row_that_scans_a_missing_path_is_reported",
    },
    GateDescriptor {
        name: "verify-rewrite-proofs",
        help: "Verify every optimizer rewrite proof fixture",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered IR corpus",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_registry::gates::verify_rewrite_proofs::tests::solver_verdicts_are_classified_soundly",
    },
    GateDescriptor {
        name: "version-matrix",
        help: "Regenerate release/evidence/version/version-matrix.json and release/evidence/version/release-tag-plan.json from \
        the workspace manifests, Cargo.lock and the release docs, and report each line the \
        committed copies disagree on. Proves every publishable crate carries the version the \
        release train declares, that every required release package is present at its expected \
        version, that pinned dependency and lockfile versions match, that no release doc gives \
        a bare tag command, and that release notes carry no stale version token. Proves nothing \
        about what is published on a registry: every fact here is read from this checkout.",
        package: "xtask",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &[
            "release/evidence/version/version-matrix.json",
            "release/evidence/version/release-tag-plan.json",
        ],
        prerequisites: &[],
        proof: "crate::release::version_matrix::tests::bare_final_tag_commands_are_rejected",
    },
    GateDescriptor {
        name: "vyre-release-gate",
        help: "Enforce release evidence closure; the default judges the prepublication set, --launch-complete judges the post-ship set, --manifest PATH names another manifest",
        package: "xtask-evidence",
        areas: &["prepublish", "release-evidence"],
        subject: "release evidence matrices",
        artifacts: &[],
        prerequisites: &[],
        proof: "xtask_evidence::release::vyre_release_gate::paths::tests::evidence_paths_that_climb_past_the_repository_root_are_rejected",
    },
    GateDescriptor {
        name: "whats-similar",
        help: "Report duplicate operations by IR shape across the whole registry; --op-id ID adds a focused view",
        package: "xtask-registry",
        areas: &["prepublish"],
        subject: "registered operations",
        artifacts: &[crate::artifact_paths::REGISTERED_OP_DUPLICATES_ARTIFACT],
        prerequisites: &[],
        proof: "xtask_registry::gates::whats_similar::query::tests::a_skip_class_over_its_ceiling_is_a_finding",
    },
    GateDescriptor {
        name: "wire-determinism",
        help: "Enforce wire-determinism contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::wire_determinism::tests::a_suite_reserves_only_what_its_own_manifest_entry_names",
    },
    GateDescriptor {
        name: "workspace-check",
        help: "Enforce workspace-check contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::workspace_build::tests::workspace_check_invokes_cargo_check_across_all_features",
    },
    GateDescriptor {
        name: "workspace-clippy",
        help: "Enforce workspace-clippy contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::workspace_build::tests::the_driver_separator_bounds_the_cargo_arguments",
    },
    GateDescriptor {
        name: "workspace-docs",
        help: "Enforce workspace-docs contracts",
        package: "xtask",
        areas: &["docs"],
        subject: "owned documentation pages",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::workspace_build::tests::workspace_docs_constructs_no_deps_doc_arguments",
    },
    GateDescriptor {
        name: "workspace-membership",
        help: "Enforce workspace-membership contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "workspace manifests",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::manifest_contract::tests::inheritance_without_an_entry_is_neither_versioned_nor_unversioned",
    },
    GateDescriptor {
        name: "workspace-tests",
        help: "Enforce workspace-tests contracts",
        package: "xtask",
        areas: &["contract-rules"],
        subject: "registered system invariants",
        artifacts: &[],
        prerequisites: &[],
        proof: "crate::gates::workspace_build::tests::workspace_tests_resolves_tested_layer_contract",
    },
];

/// All gate names declared in `GATE_METADATA`, sorted.
#[must_use]
pub fn all_gate_names() -> Vec<&'static str> {
    GATE_METADATA.iter().map(|d| d.name).collect()
}
/// Authoritative descriptor for `gate_name`, or `None` if it is not registered.
#[must_use]
pub fn descriptor(gate_name: &str) -> Option<&'static GateDescriptor> {
    GATE_METADATA.iter().find(|d| d.name == gate_name)
}

/// Authoritative descriptor for `gate_name`.
///
/// # Panics
///
/// Panics when `gate_name` is not in `GATE_METADATA`.
#[must_use]
pub fn descriptor_by_name(gate_name: &str) -> &'static GateDescriptor {
    descriptor(gate_name).unwrap_or_else(|| panic!("gate `{gate_name}` is not in GATE_METADATA"))
}

/// All gate names owned by `package`, sorted.
#[must_use]
pub fn owned_by(package: &str) -> Vec<&'static str> {
    GATE_METADATA
        .iter()
        .filter(|d| d.package == package)
        .map(|d| d.name)
        .collect()
}

/// All distinct area names declared across all gate descriptors, sorted.
#[must_use]
pub fn areas() -> Vec<&'static str> {
    let mut set = BTreeSet::new();
    for row in GATE_METADATA {
        for area in row.areas {
            set.insert(*area);
        }
    }
    set.into_iter().collect()
}

/// All gate names belonging to `area`, sorted.
#[must_use]
pub fn gates_in_area(area: &str) -> Vec<&'static str> {
    GATE_METADATA
        .iter()
        .filter(|d| d.areas.contains(&area))
        .map(|d| d.name)
        .collect()
}

/// Exact generated artifact paths with their owning gate names.
#[must_use]
pub fn generated_artifacts() -> Vec<(&'static str, &'static str)> {
    let mut pairs = Vec::new();
    for d in GATE_METADATA {
        for art in d.artifacts {
            pairs.push((*art, d.name));
        }
    }
    pairs.sort_unstable_by_key(|(art, _)| *art);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// WHY: Section 182.2 requires every gate metadata entry to be sorted by name.
    #[test]
    fn metadata_is_sorted_by_name() {
        let names: Vec<&str> = GATE_METADATA.iter().map(|d| d.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(
            names, sorted,
            "GATE_METADATA must be sorted alphabetically by name"
        );
    }

    /// WHY: Section 182.2 requires every gate metadata entry to have a non-empty name and unique entry.
    #[test]
    fn every_gate_name_is_unique() {
        let mut names = BTreeSet::new();
        for d in GATE_METADATA {
            assert!(
                names.insert(d.name),
                "duplicate gate name in GATE_METADATA: {}",
                d.name
            );
        }
    }

    /// WHY: Section 182.2.1 requires every descriptor to declare valid fields without provisional values.
    #[test]
    fn every_metadata_entry_passes_validation() {
        for d in GATE_METADATA {
            let failures = d.failures();
            assert!(
                failures.is_empty(),
                "gate `{}` has invalid metadata: {:?}",
                d.name,
                failures
            );
        }
    }

    /// WHY: Section 182.5.3 requires exact 1-to-1 generated artifact ownership without duplicate owners.
    #[test]
    fn artifact_paths_are_uniquely_owned() {
        let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
        for (artifact, gate) in generated_artifacts() {
            if let Some(existing) = owners.insert(artifact, gate) {
                panic!(
                    "artifact `{artifact}` is declared by both `{existing}` and `{gate}`: each artifact must have exactly 1 owning gate"
                );
            }
        }
    }

    /// WHY: Section 182 requires every gate descriptor in GATE_METADATA to carry a unique mutation proof identity.
    #[test]
    fn every_descriptor_proof_is_unique() {
        let mut proofs = BTreeSet::new();
        for d in GATE_METADATA {
            assert!(
                proofs.insert(d.proof),
                "duplicate proof identity in GATE_METADATA for gate `{}`: `{}`",
                d.name,
                d.proof
            );
        }
    }
}
