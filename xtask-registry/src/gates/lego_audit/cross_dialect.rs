//! Check 4: a dialect names a sibling edge in the prelude rather than reaching into it.
//!
//! Each directory under `vyre-libs/src` is one dialect. A dialect that imports
//! from inside another dialect's module tree couples the two at a path nothing
//! collects, so the set of cross-dialect edges cannot be read anywhere. An edge
//! that belongs is re-exported from `vyre_libs::prelude` and imported from
//! there.

#[allow(unused_imports)]
use super::*;

/// Walk `vyre-libs/src/<dialect>/**/*.rs` and report every `use` that reaches
/// into `crate::<other_dialect>::...` or `vyre_libs::<other_dialect>::...`.
///
/// A dialect owns its own surface and depends downward on `vyre-primitives`. An
/// edge to a sibling is allowed, and some are necessary: a linear layer is a
/// bias-matmul. What is not allowed is naming that edge from inside the
/// importing file, three levels into the other dialect's module tree, where
/// nothing collects it. The edge is declared once at the crate root, in
/// `vyre_libs::prelude`, and imported from there.
///
/// The check is structural  -  it parses Rust use trees with `syn`, so grouped
/// imports, aliases and globs are audited consistently without relying on
/// line-oriented grep. It reads paths, not visibility: a `pub use` re-export is
/// skipped because re-exporting is how the seam itself is written, and every
/// other import is judged by where it points.
///
/// `lego-quick` asks a weaker question over the same subject: whether some
/// feature gating the importing dialect enables one gating the imported
/// dialect. Feature aggregates make that true by accident, so it passes edges
/// this check reports. The two are not one measurement with two answers, and
/// collapsing them onto the stricter rule is open work.
pub(super) fn check_4_cross_dialect_reachthrough(report: &mut Report) -> usize {
    report.note("[4/10] Cross-dialect reach-through (a dialect names a sibling edge in vyre_libs::prelude, not from inside its own module tree)".to_string());
    let checkout = xtask::checkout::checkout_root();
    let libs_root = Some(checkout.join("vyre-libs").join("src"));
    let Some(libs_root) = libs_root.filter(|p| p.is_dir()) else {
        report.find(violation(
            "  ⚠ vyre-libs/src not reachable from xtask. Fix: invoke from the workspace root."
                .to_string(),
        ));
        return 0;
    };
    let (dialects, list_errors) = list_dialect_dirs(&libs_root);
    if !list_errors.is_empty() {
        for error in &list_errors {
            report.find(violation(format!("  ✗ {error}")));
        }
        return list_errors.len();
    }
    if dialects.len() < 2 {
        report.note("  ✓ fewer than 2 dialects present; nothing to cross.".to_string());
        return 0;
    }
    let mut flagged = 0usize;
    for dialect in &dialects {
        let dialect_name = dialect.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let sources = xtask::tree_walk::pruned_by(dialect, |name| {
            !xtask::tree_walk::BUILD_OUTPUT_AND_VCS.contains(&name)
        });
        for entry in sources {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    report.find(violation(format!("  ✗ {}: failed to read dialect directory: {error}. Fix: make the checked source tree fully readable.",
                        dialect.display())));
                    flagged += 1;
                    continue;
                }
            };
            let path = entry.into_path();
            if is_test_source_path(&path) {
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let text = match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    report.find(violation(format!("  ✗ {}: failed to read Rust source for reach-through audit: {error}. Fix: make the checked source tree fully readable.",
                        path.display())));
                    flagged += 1;
                    continue;
                }
            };
            let relative = path
                .strip_prefix(&checkout)
                .unwrap_or(&path)
                .display()
                .to_string();
            let Ok(file) = syn::parse_file(&text) else {
                report.find(violation(format!("  ✗ {relative}: failed to parse Rust source for reach-through audit. Fix: keep checked-in Rust source syntactically parseable.")));
                flagged += 1;
                continue;
            };
            for use_path in collect_use_paths(&file) {
                if use_path.is_public {
                    continue;
                }
                for other in &dialects {
                    let other_name = other.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if other_name == dialect_name || other_name.is_empty() {
                        continue;
                    }
                    if use_path.imports_dialect(other_name) {
                        report.find(violation(format!(
                            "  ✗ {relative} line {}: `{}` → reaches into the `{other_name}` \
                             dialect. Fix: re-export the item from vyre-libs/src/prelude.rs and \
                             import it as `crate::prelude::…`, or hoist the shared piece into \
                             vyre-primitives.",
                            use_path.line,
                            use_path.segments.join("::")
                        )));
                        flagged += 1;
                    }
                }
            }
        }
    }
    if flagged == 0 {
        report.note(
            "  ✓ every cross-dialect edge is named in vyre_libs::prelude, not reached into"
                .to_string(),
        );
    }
    flagged
}

/// Directories under `vyre-libs/src` that are shared plumbing rather than a
/// dialect, so a dialect importing from one is not a cross-dialect edge.
///
/// Only a directory can appear here, because `list_dialect_dirs` reads the
/// dialect set from the directories under `vyre-libs/src` and a single-file
/// module is never in that set to begin with. Five rows named single-file
/// modules or a path that no longer exists and were removed for that reason:
/// `region`, `tensor_ref`, `buffer_names`, `descriptor` and `test_support`.
/// `check_0_every_exemption_is_live` holds each remaining row to an existing
/// directory, so the next row that goes the same way fails instead of reading
/// as coverage.
pub(super) const SHARED_PLUMBING_DIRS: [&str; 1] = ["builder"];

/// Shared-plumbing rows that name no directory under `libs_src`.
pub(super) fn dead_plumbing_rows(libs_src: &std::path::Path) -> Vec<&'static str> {
    SHARED_PLUMBING_DIRS
        .into_iter()
        .filter(|dir| !libs_src.join(dir).is_dir())
        .collect()
}

pub(super) fn list_dialect_dirs(root: &std::path::Path) -> (Vec<std::path::PathBuf>, Vec<String>) {
    let read_dir = match std::fs::read_dir(root) {
        Ok(read_dir) => read_dir,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!(
                    "{}: failed to read dialect root: {error}. Fix: make vyre-libs/src fully readable.",
                    root.display()
                )],
            );
        }
    };
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for entry in read_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(format!(
                    "{}: failed to read dialect root entry: {error}. Fix: make vyre-libs/src fully readable.",
                    root.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if SHARED_PLUMBING_DIRS.contains(&name) {
            continue;
        }
        out.push(path);
    }
    (out, errors)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::gates::lego_audit::test_ops::{op, op_with_fingerprint};
    #[allow(unused_imports)]
    use std::path::PathBuf;

    /// WHY: the shared-plumbing list is consumed by a directory filter, so a row
    /// naming a single-file module or a path that was removed is skipped by
    /// nothing and still reads as a reviewed exemption. Five of the six rows were
    /// in that state. The check runs against a directory it is handed rather than
    /// the checkout, so both directions are proved: a row with a directory behind
    /// it is live, and one without it is reported.
    ///
    /// What this does not catch: a directory that exists but is a dialect rather
    /// than plumbing. That judgement is the reviewer's and the row carries it.
    #[test]
    fn a_plumbing_row_without_a_directory_behind_it_is_dead() {
        let libs_src = tempfile::tempdir().expect("temporary vyre-libs/src");

        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            SHARED_PLUMBING_DIRS.to_vec(),
            "every row is dead against a tree that holds none of them"
        );

        for dir in SHARED_PLUMBING_DIRS {
            std::fs::create_dir(libs_src.path().join(dir)).expect("plumbing directory");
        }
        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            Vec::<&str>::new(),
            "no row is dead once every one of them names a directory"
        );

        let first = SHARED_PLUMBING_DIRS[0];
        std::fs::remove_dir(libs_src.path().join(first)).expect("remove one plumbing directory");
        assert_eq!(
            dead_plumbing_rows(libs_src.path()),
            vec![first],
            "the row whose directory went away is the one reported"
        );
    }

}
