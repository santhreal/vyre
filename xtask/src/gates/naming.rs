//! Public operation functions follow the canonical naming scheme.
//!
//! An operation is named for what it produces, not for the act of producing it.
//! `compute_`, `do_`, `run_`, `make_`, `create_` and `new_` prefixes and `_op`,
//! `_impl` and `_internal` suffixes all describe the machinery instead of the
//! result, and two of them arrive as a pair when someone wraps an existing
//! function rather than editing it.
//!
//! The non-snake-case check is belt and braces. rustc rejects a PascalCase free
//! function under `non_snake_case` already, so this only catches one smuggled in
//! behind an `allow`.

use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Prefixes that name the act rather than the result.
const BANNED_PREFIXES: &[&str] = &["compute_", "do_", "run_", "make_", "create_", "new_"];

/// Suffixes that name the machinery rather than the result.
const BANNED_SUFFIXES: &[&str] = &["_op", "_impl", "_internal"];

/// Filenames inside the operation tree that are not operation sources.
const NOT_OPERATION_SOURCES: &[&str] = &[
    "mod.rs",
    "lib.rs",
    "builder.rs",
    "harness.rs",
    "relation_analyzer.rs",
];

/// An operation is named for what it computes, not for the fact that it runs.
pub struct OperationNames;

impl Gate for OperationNames {
    fn name(&self) -> &'static str {
        "operation-names"
    }

    fn help(&self) -> &'static str {
        "public operation functions named for the act rather than the result"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let files: Vec<PathBuf> = tree
            .rust(&["vyre-libs/src"])?
            .into_iter()
            .filter(|path| is_operation_source(path))
            .collect();
        report.note(format!("scanned {} operation source file(s)", files.len()));
        for file in &files {
            let text = tree.read(file)?;
            for (number, line) in crate::gates::scan::numbered(&text) {
                let Some(function) = public_function_name(line) else {
                    continue;
                };
                if let Some(prefix) = BANNED_PREFIXES
                    .iter()
                    .find(|prefix| function.starts_with(**prefix))
                {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("`{function}` opens with the banned prefix `{prefix}`"),
                        "name the function for the value it returns",
                    ));
                }
                if let Some(suffix) = BANNED_SUFFIXES
                    .iter()
                    .find(|suffix| function.ends_with(**suffix))
                {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("`{function}` ends with the banned suffix `{suffix}`"),
                        "name the function for the value it returns; a wrapper suffix means \
                         two functions where one belongs",
                    ));
                }
                if function.chars().any(char::is_uppercase) {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("`{function}` is not snake case"),
                        "rename to snake case; an allow(non_snake_case) is how this reaches \
                         a reviewer at all",
                    ));
                }
            }
        }
        Ok(report)
    }
}

/// Whether a file inside the operation tree holds operations.
///
/// The exclusion is by filename, which is the weakness the shell original had
/// too: a new module-root or helper filename is scanned as an operation source
/// until someone adds it here.
fn is_operation_source(path: &Path) -> bool {
    if path.to_string_lossy().contains("/tests/") {
        return false;
    }
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    !NOT_OPERATION_SOURCES.contains(&name)
}

/// The name of a public free function declared at the top level of a file.
///
/// Only column zero counts, as in the rule this replaces: an indented `pub fn`
/// is an inherent or trait method and is named by its type, not by this scheme.
fn public_function_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("pub fn ")?;
    let end = rest
        .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() {
        return None;
    }
    // A generic or parenthesised continuation is what follows a real
    // declaration. Anything else is prose that happens to start this way.
    let tail = rest[end..].trim_start();
    if tail.starts_with('(') || tail.starts_with('<') {
        Some(name)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the shell rule anchored at column zero, so an indented method was
    /// never in scope. A predicate that trimmed first would report every method
    /// in the crate and the pin would be meaningless.
    #[test]
    fn only_a_top_level_public_function_is_in_scope() {
        assert_eq!(public_function_name("pub fn fuse(a: u32) -> u32 {"), Some("fuse"));
        assert_eq!(
            public_function_name("pub fn reduce<T>(values: &[T]) {"),
            Some("reduce")
        );
        assert_eq!(public_function_name("    pub fn method(&self) {"), None);
        assert_eq!(public_function_name("// pub fn quoted()"), None);
        assert_eq!(public_function_name("pub fn"), None);
    }

    /// WHY: the exclusion list is by filename and that is exactly how a new
    /// helper file starts being read as an operation source. The test pins the
    /// five names the rule knows about so adding a sixth is a deliberate edit.
    #[test]
    fn module_roots_and_helpers_are_not_operation_sources() {
        assert!(is_operation_source(Path::new("vyre-libs/src/graph/fuse.rs")));
        assert!(!is_operation_source(Path::new("vyre-libs/src/graph/mod.rs")));
        assert!(!is_operation_source(Path::new("vyre-libs/src/lib.rs")));
        assert!(!is_operation_source(Path::new("vyre-libs/src/graph/builder.rs")));
        assert!(!is_operation_source(Path::new(
            "vyre-libs/src/tests/fuse_contracts.rs"
        )));
    }
}
