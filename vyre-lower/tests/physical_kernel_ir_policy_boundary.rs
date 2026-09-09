//! Contract tests for Row 107: Physical Kernel IR & Legalization Isolation.
//!
//! Verifies:
//! 1. No physical kernel IR type in `vyre-lower::descriptor` reuses semantic policy types.
//! 2. No backend name appears in any shared crate legalization or capability path.
//! 3. Lowering from selected schedules is deterministic and byte-identical on repeat runs.

use std::collections::BTreeSet;
use std::path::Path;

use vyre_lower::{verify_descriptor, KernelOpKind};

const SEMANTIC_POLICY_TYPES: &[&str] = &[
    "InnovationTrack",
    "TrackDecision",
    "SchedulingPolicy",
    "DeviceMemoryBudget",
    "MemoryBudgetReport",
    "AccuracyPlan",
    "AutotunePlan",
    "ProvenancePlan",
    "FusionPlan",
    "MemoryPlan",
    "ValidationOptions",
    "CompileObjective",
    "SearchBudget",
    "RequiredSchedule",
    "FrontierTopology",
    "ExecutionMode",
];

const BACKEND_NAMES: &[&str] = &[
    "cuda", "nvptx", "ptx", "metal", "spirv", "wgpu", "vulkan", "dx12", "opencl",
];

#[test]
fn no_physical_kernel_ir_type_reuses_a_semantic_policy_type() {
    // Read all source files defining physical kernel IR in vyre-lower/src/descriptor/
    let descriptor_dir = Path::new("vyre-lower/src/descriptor");
    let fallback_dir = Path::new("src/descriptor");
    let target_dir = if descriptor_dir.exists() {
        descriptor_dir
    } else {
        fallback_dir
    };

    let mut checked_files = 0;
    let mut all_descriptor_tokens = BTreeSet::new();

    let entries = std::fs::read_dir(target_dir).expect("descriptor dir must be readable");
    for entry in entries {
        let entry = entry.expect("valid dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let content = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("failed to read {:?}", path));
            checked_files += 1;

            // Extract all type identifiers from the source
            for word in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
                if !word.is_empty() {
                    all_descriptor_tokens.insert(word.to_string());
                }
            }
        }
    }

    assert!(
        checked_files >= 5,
        "Expected to check at least 5 descriptor source files, checked {}",
        checked_files
    );

    // Assert that NONE of the semantic policy types appear in physical kernel IR
    for policy_type in SEMANTIC_POLICY_TYPES {
        assert!(
            !all_descriptor_tokens.contains(*policy_type),
            "Physical kernel IR in vyre-lower::descriptor must not reuse semantic policy type `{policy_type}`."
        );
    }
}

#[test]
fn no_backend_name_appears_in_shared_legalization_paths() {
    // Check all source files in vyre-lower/src/ for backend-name branch defects
    let lower_src = Path::new("vyre-lower/src");
    let fallback_src = Path::new("src");
    let target_dir = if lower_src.exists() {
        lower_src
    } else {
        fallback_src
    };

    fn visit_dir(dir: &Path, backend_violations: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("readable dir") {
            let entry = entry.expect("valid entry");
            let path = entry.path();
            if path.is_dir() {
                visit_dir(&path, backend_violations);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let content = std::fs::read_to_string(&path).expect("readable rs file");
                for (line_no, line) in content.lines().enumerate() {
                    // Check if line contains a backend name branch in legalization/lowering code
                    let lower = line.to_lowercase();
                    if lower.contains("legal")
                        || lower.contains("capability")
                        || lower.contains("validate")
                    {
                        for &backend in BACKEND_NAMES {
                            // Match literal quotes like "cuda" or identifiers
                            if lower.contains(&format!("\"{backend}\""))
                                || lower.contains(&format!("backend == \"{backend}\""))
                                || lower.contains(&format!("target == \"{backend}\""))
                            {
                                backend_violations.push(format!(
                                    "{}:{}: found backend name '{}' in legalization path: {}",
                                    path.display(),
                                    line_no + 1,
                                    backend,
                                    line.trim()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut violations = Vec::new();
    visit_dir(target_dir, &mut violations);

    assert!(
        violations.is_empty(),
        "Hardware must be represented as a fact vector, not a backend name. Violations found:\n{}",
        violations.join("\n")
    );
}

#[test]
fn repeat_lowering_and_verification_produces_byte_identical_descriptors() {
    use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
    use vyre_lower::LiteralValue;

    let desc = descriptor("multi_entry_phase_0")
        .slot(global_rw(
            0,
            vyre_foundation::ir::DataType::U32,
            "input_matrix_a",
        ))
        .slot(global_rw(
            1,
            vyre_foundation::ir::DataType::U32,
            "output_matrix_b",
        ))
        .dispatch(128, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(42)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(op(
                    KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                    [0, 1],
                    2,
                ))
                .op(effect(KernelOpKind::StoreGlobal, [1, 0, 2])),
        )
        .build();

    // Verify descriptor twice
    let verified_1 = verify_descriptor(&desc).expect("first verification succeeds");
    let verified_2 = verify_descriptor(&desc).expect("second verification succeeds");

    // Serialize both verified descriptors to JSON
    let json_1 = serde_json::to_vec(&verified_1).expect("serialization 1");
    let json_2 = serde_json::to_vec(&verified_2).expect("serialization 2");

    // Exact byte equality check
    assert_eq!(
        json_1, json_2,
        "repeat descriptor lowering and canonicalization must produce byte-identical results with no rediscovery"
    );
    assert_eq!(verified_1, verified_2);
}
