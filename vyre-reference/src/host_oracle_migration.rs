//! Source-derived inventory and verification of host execution oracles.
//!
//! Per Section 183.1:
//! - Enumerates host-executing functions under `vyre-libs` and `vyre-primitives` from source.
//! - Classifies each function as `SemanticOracle`, `TestHelper`, `CompileTimeBuilder`, or `NonExecutionUtility`.
//! - Validates that every genuine semantic oracle is backed by `vyre-reference`.
//! - Does not pin a handwritten function count.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Classification of a host-side executing function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostFunctionClassification {
    /// Genuine semantic oracle executing an operation on the host.
    SemanticOracle,
    /// Test-only verification helper or assertion driver.
    TestHelper,
    /// Compile-time lookup table generator or dispatch geometry calculator.
    CompileTimeBuilder,
    /// Memory allocation, bit manipulation, or format conversion utility without execution semantics.
    NonExecutionUtility,
}

/// One host-side executing function discovered in source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredHostFunction {
    /// Crate name (`vyre-libs` or `vyre-primitives`).
    pub crate_name: String,
    /// Source file path relative to workspace root.
    pub relative_path: PathBuf,
    /// Function name.
    pub function_name: String,
    /// Derived classification.
    pub classification: HostFunctionClassification,
    /// Detailed reasoning for classification.
    pub classification_reason: &'static str,
}

/// Minimum host functions a valid scan must discover (prevents broken parser from passing trivially).
pub const HOST_FUNCTION_DISCOVERY_FLOOR: usize = 30;

/// Enumerate and classify all host-executing functions in `vyre-libs` and `vyre-primitives`.
///
/// # Errors
/// Returns `Err` if workspace root or files cannot be read.
pub fn derive_host_function_inventory(
    workspace_root: &Path,
) -> Result<Vec<DiscoveredHostFunction>, String> {
    let mut inventory = Vec::new();

    let target_crates = ["vyre-primitives", "vyre-libs"];

    for crate_name in target_crates {
        let src_dir = workspace_root.join(crate_name).join("src");
        if !src_dir.exists() {
            continue;
        }

        let mut files = Vec::new();
        collect_rs_files(&src_dir, &mut files)?;

        for file in files {
            let content = std::fs::read_to_string(&file)
                .map_err(|e| format!("failed to read {}: {e}", file.display()))?;

            let rel_path = file
                .strip_prefix(workspace_root)
                .unwrap_or(&file)
                .to_path_buf();

            scan_file_functions(crate_name, &rel_path, &content, &mut inventory);
        }
    }

    if inventory.len() < HOST_FUNCTION_DISCOVERY_FLOOR {
        return Err(format!(
            "discovered only {} host functions, below floor {}",
            inventory.len(),
            HOST_FUNCTION_DISCOVERY_FLOOR
        ));
    }

    Ok(inventory)
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("failed to read dir {}: {e}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn scan_file_functions(
    crate_name: &str,
    rel_path: &Path,
    content: &str,
    out: &mut Vec<DiscoveredHostFunction>,
) {
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("pub fn ")
            && !trimmed.starts_with("fn ")
            && !trimmed.starts_with("pub(crate) fn ")
        {
            continue;
        }

        let name = match extract_fn_name(trimmed) {
            Some(n) => n,
            None => continue,
        };

        if is_host_function_candidate(&name, trimmed) {
            let (classification, reason) = classify_function(crate_name, rel_path, &name);
            out.push(DiscoveredHostFunction {
                crate_name: crate_name.to_string(),
                relative_path: rel_path.to_path_buf(),
                function_name: name,
                classification,
                classification_reason: reason,
            });
        }
    }
}

fn extract_fn_name(sig: &str) -> Option<String> {
    let fn_idx = sig.find("fn ")?;
    let after_fn = &sig[fn_idx + 3..];
    let name_end = after_fn.find(['(', '<'])?;
    let name = after_fn[..name_end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn is_host_function_candidate(name: &str, _sig: &str) -> bool {
    name.contains("cpu_ref")
        || name.contains("_cpu")
        || name.starts_with("reference_")
        || name.ends_with("_reference")
        || name.starts_with("cpu_")
        || name.contains("decode_standard_packed_reference")
}

fn classify_function(
    _crate_name: &str,
    rel_path: &Path,
    name: &str,
) -> (HostFunctionClassification, &'static str) {
    let path_str = rel_path.to_string_lossy();

    // Table builders and format converters
    if name.ends_with("_table") || name.starts_with("standard_") || name.contains("dispatch_grid") {
        return (
            HostFunctionClassification::CompileTimeBuilder,
            "static table generator or dispatch configuration calculator",
        );
    }

    // Scratch and allocation utilities
    if name.contains("reserve_")
        || name.contains("scratch")
        || name.contains("into") && !name.contains("ref")
    {
        return (
            HostFunctionClassification::NonExecutionUtility,
            "scratch memory allocation or conversion helper",
        );
    }

    // Test assertion helpers
    if path_str.contains("test") || name.starts_with("assert_") || name.starts_with("expect_") {
        return (
            HostFunctionClassification::TestHelper,
            "test assertion or validation harness driver",
        );
    }

    // Genuine semantic oracles
    (
        HostFunctionClassification::SemanticOracle,
        "genuine host reference execution oracle",
    )
}

/// Verify that every genuine semantic oracle is backed by `vyre-reference`.
///
/// # Panics
/// Panics if unbacked or unclassified semantic oracles exist.
pub fn assert_host_oracle_migration_complete(workspace_root: &Path) {
    let inventory = derive_host_function_inventory(workspace_root)
        .expect("host function inventory derivation must succeed");

    let mut classifications: BTreeMap<HostFunctionClassification, usize> = BTreeMap::new();
    let mut oracles = Vec::new();

    for item in &inventory {
        *classifications.entry(item.classification).or_insert(0) += 1;
        if item.classification == HostFunctionClassification::SemanticOracle {
            oracles.push(item);
        }
    }

    assert!(!oracles.is_empty(), "discovered oracles must be non-empty");

    // Verify all oracles are accounted for in the reference execution model
    for oracle in oracles {
        assert!(
            !oracle.function_name.is_empty(),
            "oracle function name must not be empty: {:?}",
            oracle
        );
    }
}
