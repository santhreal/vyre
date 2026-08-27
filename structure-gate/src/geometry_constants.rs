//! Structural gate: operations in `vyre-libs` and `vyre-primitives` must not
//! declare hardcoded execution geometry constants (workgroup sizes, block
//! lanes, invocation counts, tile sizes, elements-per-invocation constants,
//! or stage counts).
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

/// Collect all banned geometry constant declarations in `vyre-libs/src` and `vyre-primitives/src`.
#[must_use]
pub fn geometry_constant_failures(root: &Path) -> Vec<String> {
    let mut failures = Vec::new();

    for source in rust_sources_with_text(root) {
        let (file, text) = match source {
            SourceText::Read { path, text } => (path, text),
            SourceText::Unread { path, reason } => {
                if !is_test_source(&path)
                    && (path.starts_with("vyre-libs/src/")
                        || path.starts_with("vyre-primitives/src/"))
                {
                    failures.push(format!(
                        "{path} was not read: {reason}; the geometry constant gate cannot report an unread source as clean"
                    ));
                }
                continue;
            }
        };
        if is_test_source(&file) {
            continue;
        }
        if !file.starts_with("vyre-libs/src/") && !file.starts_with("vyre-primitives/src/") {
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

    #[test]
    fn mutation_gate_reintroduced_constant_is_caught() {
        let synthetic_source = "pub const BLOCK_LANES: u32 = 1024;\nfn foo() {}\n";
        let mut found = false;
        for line in synthetic_source.lines() {
            if is_ident_defined_on_line(line.trim(), "BLOCK_LANES") {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Reintroduced geometry constant must be caught by gate"
        );
    }
}
