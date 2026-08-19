//! Record types, constants, and path helpers for host oracle detection.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub(super) const TARGET_ROOTS: &[&str] =
    &["vyre-libs/src", "vyre-primitives/src", "vyre-driver/src"];

/// Exact canonical qualified IR builder and operation types representing AST/IR owners.
pub(super) const EXACT_CANONICAL_IR_BUILDER_PATHS: &[&str] = &[
    "vyre_foundation::ir::Program",
    "vyre_foundation::ir::Node",
    "vyre_foundation::ir::Expr",
    "vyre_foundation::operation::OperationRegistration",
    "vyre_foundation::operation::SemanticOperation",
    "vyre_foundation::operation::OperationRegistry",
    "vyre_primitives::hardware::HardwareEntry",
];

/// Exact canonical qualified dispatcher capability traits and types derived from actual source imports.
pub(super) const EXACT_CANONICAL_DISPATCHER_PATHS: &[&str] =
    &["vyre_foundation::program_dispatch::ProgramDispatcher"];

/// Exact terminal scalar types.
pub(super) const SCALAR_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool",
];

/// Corrective action for a production host oracle finding.
pub(super) const FIX: &str = "move the host reference implementation into a #[cfg(test)] module or vyre-reference, \
                   or replace dynamic registration evaluation with exact byte fixtures; production code \
                   in shipping crates must not execute host CPU mathematical oracles or unisolated semantic twins";

/// Structural record of a parameter in a function signature.
#[derive(Clone, Debug)]
pub(super) struct FunctionParamRecord {
    pub(super) name: String,
    pub(super) qualified_custom_types: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub(super) struct ParamCalleeFlow {
    pub(super) param_idx: usize,
    pub(super) callee_name: String,
    pub(super) callee_arg_idx: usize,
}

#[derive(Clone, Debug)]
pub(super) struct FunctionRecord {
    pub(super) name: String,
    pub(super) file: PathBuf,
    pub(super) module_path: Vec<String>,
    pub(super) line: u32,
    pub(super) is_public: bool,
    pub(super) is_test_scoped: bool,
    pub(super) is_ir_builder: bool,
    pub(super) is_gpu_dispatch_root: bool,
    pub(super) is_data_processing: bool,
    pub(super) is_wire_codec: bool,
    pub(super) is_sizing_or_validator: bool,
    pub(super) returns_data_output: bool,
    pub(super) is_explicit_oracle_name: bool,
    pub(super) has_canonical_dispatcher_param: bool,
    pub(super) param_custom_types: BTreeSet<String>,
    pub(super) return_custom_types: BTreeSet<String>,
    pub(super) has_collection_payload_inputs: bool,
    pub(super) params: Vec<FunctionParamRecord>,
    pub(super) direct_dispatched_param_indices: BTreeSet<usize>,
    pub(super) param_callee_flows: Vec<ParamCalleeFlow>,
    pub(super) stages_semantic_resident_upload: bool,
}

/// Structural record of a function call or reference site.
#[derive(Clone, Debug)]
pub(super) struct CallSiteRecord {
    pub(super) callee: String,
    pub(super) caller_file: PathBuf,
    pub(super) caller_module: Vec<String>,
    pub(super) caller_fn_idx: Option<usize>,
    pub(super) line: u32,
    pub(super) is_method_call: bool,
    pub(super) is_in_test: bool,
    pub(super) is_in_expected_output: bool,
    pub(super) is_in_fallback: bool,
    pub(super) is_in_post_dispatch: bool,
    pub(super) is_in_op_reg: bool,
}

/// Structural record of a static or const definition in source code.
#[derive(Clone, Debug)]
pub(super) struct StaticConstRecord {
    pub(super) name: String,
    pub(super) file: PathBuf,
    pub(super) module_path: Vec<String>,
    pub(super) line: u32,
    pub(super) is_test_scoped: bool,
    pub(super) has_semantic_operation: bool,
}

/// Derive base module path from file path relative to crate `src/` directory.
pub(super) fn base_module_path(path: &Path) -> Vec<String> {
    let path_str = path.to_string_lossy();
    let relative = if let Some(idx) = path_str.find("/src/") {
        &path_str[idx + 5..]
    } else if let Some(idx) = path_str.find("src/") {
        &path_str[idx + 4..]
    } else {
        path_str.as_ref()
    };

    let without_ext = relative.strip_suffix(".rs").unwrap_or(relative);
    let segments: Vec<&str> = without_ext.split('/').collect();

    let mut mod_path = Vec::new();
    for &seg in &segments {
        if seg == "mod" || seg == "lib" || seg == "main" {
            continue;
        }
        mod_path.push(seg.to_string());
    }
    mod_path
}
pub(super) fn normalize_qualified_path(path: &str) -> String {
    let path = path.trim();
    if let Some(rest) = path.strip_prefix("crate::") {
        rest.to_string()
    } else if let Some(idx) = path.find("::") {
        let first = &path[..idx];
        if first.starts_with("vyre_") {
            path[idx + 2..].to_string()
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    }
}

/// Flatten a `syn::UseTree` into mapping of local name -> fully-qualified imported path.
pub(super) fn extract_use_tree(
    prefix: &str,
    tree: &syn::UseTree,
    out: &mut BTreeMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let new_prefix = if prefix.is_empty() {
                p.ident.to_string()
            } else {
                format!("{prefix}::{}", p.ident)
            };
            extract_use_tree(&new_prefix, &p.tree, out);
        }
        syn::UseTree::Name(n) => {
            let full_path = if prefix.is_empty() {
                n.ident.to_string()
            } else {
                format!("{prefix}::{}", n.ident)
            };
            out.insert(n.ident.to_string(), full_path);
        }
        syn::UseTree::Rename(r) => {
            let full_path = if prefix.is_empty() {
                r.ident.to_string()
            } else {
                format!("{prefix}::{}", r.ident)
            };
            out.insert(r.rename.to_string(), full_path);
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                extract_use_tree(prefix, item, out);
            }
        }
        syn::UseTree::Glob(_) => {
            // Globs do not provide explicit single-ident bindings and fail closed for root qualification
        }
    }
}
