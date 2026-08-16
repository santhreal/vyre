//! The gate that holds public op functions to the canonical naming scheme.
//!
//! This was `scripts/check_op_names.sh`, wired into CI only through the
//! `check-cat-a` composite. A shell script that asserts an invariant is a gate
//! with no baseline, no place in the sweep and no report a caller can count, so
//! the rules moved here and the script is gone.
//!
//! The matching is deliberately the same shape as the three regular expressions
//! the script held, judged against `pub fn` declarations only, so a name the
//! script rejected is a name this rejects.

use std::path::Path;

use walkdir::WalkDir;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Verbs that say nothing about what an op computes.
const BANNED_PREFIXES: &[&str] = &["compute_", "do_", "run_", "make_", "create_", "new_"];

/// Suffixes that describe the implementation rather than the operation.
const BANNED_SUFFIXES: &[&str] = &["_op", "_impl", "_internal"];

/// Files whose public functions are not ops.
///
/// A module root re-exports, a builder assembles, a harness drives a test, and
/// the relation analyzer is an analysis pass. None of them name an operation.
const EXEMPT_FILE_NAMES: &[&str] = &[
    "mod.rs",
    "lib.rs",
    "builder.rs",
    "harness.rs",
    "relation_analyzer.rs",
];

/// The corrective action for every name this gate rejects.
const FIX: &str =
    "name the function for the operation it performs, in snake_case, with no implementation suffix";

/// Largest source file this gate will read.
const MAX_SOURCE_BYTES: u64 = 2_097_152;

/// The identifier a `pub fn` line declares, or `None` when the line declares no
/// function.
fn declared_function(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("pub fn ")?;
    let end = rest
        .find(|character: char| !character.is_alphanumeric() && character != '_')
        .unwrap_or(rest.len());
    let name = &rest[..end];
    (!name.is_empty()).then_some(name)
}

/// Every naming rule `name` breaks.
fn violations(name: &str) -> Vec<String> {
    let mut found = Vec::new();
    if let Some(prefix) = BANNED_PREFIXES
        .iter()
        .find(|prefix| name.starts_with(**prefix))
    {
        found.push(format!("`{name}` opens with the banned prefix `{prefix}`"));
    }
    if let Some(suffix) = BANNED_SUFFIXES
        .iter()
        .find(|suffix| name.ends_with(**suffix))
    {
        found.push(format!("`{name}` ends with the banned suffix `{suffix}`"));
    }
    // Rust rejects a PascalCase free function under `non_snake_case`, so this
    // rule only fires where an `allow` was added to silence it.
    if name.chars().any(char::is_uppercase) {
        found.push(format!("`{name}` is not snake_case"));
    }
    found
}

/// Whether this file's public functions are ops.
fn is_op_source(path: &Path) -> bool {
    if path.extension().is_none_or(|extension| extension != "rs") {
        return false;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if EXEMPT_FILE_NAMES.contains(&name) {
        return false;
    }
    !path
        .components()
        .any(|component| component.as_os_str() == "tests")
}

/// Holds every public op function in `vyre-libs` to the canonical naming scheme.
pub struct OpNames;

impl Gate for OpNames {
    fn name(&self) -> &'static str {
        "op-names"
    }

    fn help(&self) -> &'static str {
        "Hold every public function in vyre-libs op sources to the canonical operation naming scheme"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let libs = ctx.root.join("vyre-libs/src");
        if !libs.is_dir() {
            return Err(GateError::new(
                format!("{} is not a directory", libs.display()),
                "run the gate against a checkout that contains vyre-libs",
            ));
        }
        let mut report = Report::clean();
        let mut scanned = 0usize;
        for entry in WalkDir::new(&libs) {
            let entry = entry.map_err(|error| {
                GateError::new(
                    format!("cannot walk {}: {error}", libs.display()),
                    "make every directory under vyre-libs/src readable",
                )
            })?;
            let path = entry.path();
            if !entry.file_type().is_file() || !is_op_source(path) {
                continue;
            }
            let text = crate::output_arg::read_text_bounded(path, MAX_SOURCE_BYTES, "op-name scan")
                .map_err(|error| {
                    GateError::new(
                        format!("cannot read {}: {error}", path.display()),
                        "make the file readable, or split it under the scan bound",
                    )
                })?;
            scanned += 1;
            for (index, line) in text.lines().enumerate() {
                // The script matched `^pub fn`, so an inherent method stays out
                // of scope: a method is named against its type, not the op.
                if !line.starts_with("pub fn ") {
                    continue;
                }
                let Some(name) = declared_function(line) else {
                    continue;
                };
                let line_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                for violation in violations(name) {
                    report.find(Finding::at(
                        path.strip_prefix(&ctx.root).unwrap_or(path),
                        line_number,
                        violation,
                        FIX,
                    ));
                }
            }
        }
        report.note(format!("scanned {scanned} op source file(s)"));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the ported rules are the reason this gate exists, so each of the
    /// three the script held has to fire, and a canonical name has to pass.
    #[test]
    fn rejects_every_banned_shape_and_accepts_a_canonical_name() {
        assert_eq!(violations("matmul_f32").len(), 0);
        assert_eq!(violations("compute_matmul").len(), 1);
        assert_eq!(violations("matmul_impl").len(), 1);
        assert_eq!(violations("matMul").len(), 1);
        assert_eq!(violations("do_matMul_op").len(), 3);
    }

    /// WHY: the identifier is what the message names, so a `pub fn` with
    /// generics or a return type must still yield the bare name.
    #[test]
    fn reads_the_identifier_out_of_a_declaration() {
        assert_eq!(
            declared_function("pub fn matmul_f32<T: Copy>(a: &[T]) -> Vec<T> {"),
            Some("matmul_f32")
        );
        assert_eq!(declared_function("    let x = 1;"), None);
    }

    /// WHY: a module root and a test tree are exempt, and exempting them by
    /// substring would also exempt a real op source whose path contains the
    /// word.
    #[test]
    fn scopes_the_scan_to_op_sources() {
        assert!(is_op_source(Path::new("vyre-libs/src/geom/rotate.rs")));
        assert!(!is_op_source(Path::new("vyre-libs/src/geom/mod.rs")));
        assert!(!is_op_source(Path::new(
            "vyre-libs/src/geom/tests/rotate.rs"
        )));
        assert!(is_op_source(Path::new("vyre-libs/src/geom/testable.rs")));
        assert!(!is_op_source(Path::new("vyre-libs/src/geom/rotate.md")));
    }
}
