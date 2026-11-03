//! Structural gate: operations in the `vyre-libs` family and `vyre-primitives`
//! must not declare hardcoded execution geometry constants (workgroup sizes,
//! block lanes, invocation counts, tile sizes, elements-per-invocation
//! constants, or stage counts).
//!
//! Geometry is selected by `vyre-megakernel` from `GeometryRequirements` and
//! the widths a target profile admits, not declared as a constant in library
//! operations.

use std::path::Path;

use crate::backend_vocabulary::is_test_source;
use crate::source_scan::{rust_sources_with_text, SourceText};

const BANNED_PATTERNS: &[&str] = &[
    "BLOCK_LANES",
    "WORKGROUP_LANES",
    "PORTABLE_WORKGROUP_INVOCATIONS",
    "SCAN_WORKGROUP_LANES",
    "FRONTIER_WORD_SCAN_BLOCK_LANES",
    "FRONTIER_TO_QUEUE_WORKGROUP_LANES",
    "REDUCE_MEAN_TILE",
    "REDUCE_VARIANCE_TILE",
    "DOT_TILE",
    "SOFTMAX_TILE",
    "LAYER_NORM_TILE",
    "RMS_TILE",
    "CROSS_ENTROPY_TILE",
    "MLP_WORKGROUP",
    "ELEMENTS_PER_INVOCATION",
    "PIPELINE_STAGES",
];

/// Whether `path` is production source of a crate this gate judges: any
/// Category A crate (the `vyre-libs` facade and every `vyre-libs-<domain>`
/// partition) or the Category C crate.
fn judged_source(path: &str) -> bool {
    let Some((member, rest)) = path.split_once('/') else {
        return false;
    };
    rest.starts_with("src/")
        && (crate::workspace_rules::is_category_a_crate(member)
            || member == crate::workspace_rules::CATEGORY_C_CRATE)
}

/// Collect every banned geometry constant declared under `src/` of a
/// Category A crate or the Category C crate.
#[must_use]
pub fn geometry_constant_failures(root: &Path) -> Vec<String> {
    let mut failures = Vec::new();

    for source in rust_sources_with_text(root) {
        let (file, text) = match source {
            SourceText::Read { path, text } => (path, text),
            SourceText::Unread { path, reason } => {
                if !is_test_source(&path) && judged_source(&path) {
                    failures.push(format!(
                        "{path} was not read: {reason}; the geometry constant gate cannot report an unread source as clean"
                    ));
                }
                continue;
            }
        };
        if is_test_source(&file) || !judged_source(&file) {
            continue;
        }

        for (line_idx, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            // Skip comments and non-const declarations
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
                continue;
            }
            if !trimmed.contains("const ") {
                continue;
            }

            for &banned in BANNED_PATTERNS {
                if is_ident_defined_on_line(trimmed, banned) {
                    failures.push(format!(
                        "{file}:{} declares banned geometry constant `{banned}`; \
                         operations declare `GeometryRequirements` and let `vyre-megakernel` select \
                         the geometry rather than hardcoding it",
                        line_idx + 1
                    ));
                }
            }
        }
    }

    failures
}

fn is_ident_defined_on_line(line: &str, ident: &str) -> bool {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "const" {
            continue;
        }
        return tokens.next().and_then(|name| name.split(':').next()) == Some(ident);
    }
    false
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_banned_constant_declarations() {
        assert!(is_ident_defined_on_line(
            "pub const BLOCK_LANES: u32 = 1024;",
            "BLOCK_LANES"
        ));
        assert!(is_ident_defined_on_line(
            "const REDUCE_MEAN_TILE: u32 = 256;",
            "REDUCE_MEAN_TILE"
        ));
        assert!(is_ident_defined_on_line(
            "const DOT_TILE: u32 = 256;",
            "DOT_TILE"
        ));
        assert!(is_ident_defined_on_line(
            "pub(crate) const PORTABLE_WORKGROUP_INVOCATIONS: u32 = 256;",
            "PORTABLE_WORKGROUP_INVOCATIONS"
        ));

        // Non-definitions / uses / comments should not match
        assert!(!is_ident_defined_on_line(
            "let lanes = BLOCK_LANES;",
            "BLOCK_LANES"
        ));
        assert!(!is_ident_defined_on_line(
            "fn use_block_lanes(size: u32) {}",
            "BLOCK_LANES"
        ));
        assert!(!is_ident_defined_on_line(
            "const OTHER_CONSTANT: u32 = 123;",
            "BLOCK_LANES"
        ));
        assert!(!is_ident_defined_on_line(
            "pub const SOFT_MAX_N: u32 = PORTABLE_WORKGROUP_INVOCATIONS * PORTABLE_WORKGROUP_INVOCATIONS;",
            "PORTABLE_WORKGROUP_INVOCATIONS"
        ));
    }

    /// WHY: the scope named `vyre-libs/src/`, and the domain split moved every
    /// operation into `vyre-libs-<domain>/src/`, so the gate judged a facade
    /// with one `lib.rs` and passed over the tree it exists for. The members
    /// are read from the workspace manifest, so a partition added later is
    /// judged without editing this test. It does not catch a banned name
    /// missing from `BANNED_PATTERNS`.
    #[test]
    fn every_composition_and_intrinsic_crate_is_in_scope() {
        let root = crate::workspace_manifest::workspace_root();
        let members = crate::workspace_manifest::workspace_members(&root);
        let mut judged_partitions = 0usize;
        for member in &members {
            let name = member.rsplit('/').next().unwrap_or(member);
            let expected = crate::workspace_rules::is_category_a_crate(name)
                || name == crate::workspace_rules::CATEGORY_C_CRATE;
            assert_eq!(
                judged_source(&format!("{member}/src/lib.rs")),
                expected,
                "Fix: `{member}` is {}a composition or intrinsic crate, so its source is {}judged",
                if expected { "" } else { "not " },
                if expected { "" } else { "not " },
            );
            if expected && name.starts_with("vyre-libs-") {
                judged_partitions += 1;
            }
        }
        assert!(
            judged_partitions > 0,
            "Fix: the workspace declares no `vyre-libs-<domain>` member, so this test proves nothing about the split"
        );
        assert!(!judged_source("vyre-libs-nn/tests/case.rs"));
    }
}
