//! Check 4: a dialect names a sibling edge in the prelude rather than reaching into it.
//!
//! Each directory under `vyre-libs/src` is one dialect. A dialect that imports
//! from inside another dialect's module tree couples the two at a path nothing
//! collects, so the set of cross-dialect edges cannot be read anywhere. An edge
//! that belongs is re-exported from `vyre_libs::prelude` and imported from
//! there.

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
                    if is_substrate_target(other_name) {
                        continue;
                    }
                    if use_path.imports_dialect(other_name) {
                        report.find(violation(format!(
                            "  ✗ {relative} line {}: `{}` → reaches into the `{other_name}` \
                             dialect. Fix: re-export the item from vyre-libs/src/prelude.rs and \
                             import it as `crate::prelude::…`, or move the shared piece down into \
                             the kernel substrate the dialects compose from.",
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
/// `check_0_every_exemption_is_live` holds each remaining row to a directory
/// that still carries Rust source, so the next row that goes the same way fails
/// instead of reading as coverage.
pub(super) const SHARED_PLUMBING_DIRS: [&str; 1] = ["builder"];

/// Shared-plumbing rows whose directory under `libs_src` carries no Rust source.
pub(super) fn dead_plumbing_rows(libs_src: &std::path::Path) -> Vec<&'static str> {
    SHARED_PLUMBING_DIRS
        .into_iter()
        .filter(|dir| !structure_gate::source_scan::carries_rust_source(&libs_src.join(dir)))
        .collect()
}

/// Directories under `vyre-libs/src` that hold the kernel substrate: the
/// composition domains every dialect is built out of. A dialect naming one is
/// composing, not reaching, so an edge into these is not a cross-dialect edge.
///
/// These are the domains that used to sit in `vyre-primitives` and were reached
/// as `vyre_primitives::<domain>::…`. The path is now `crate::<domain>::…`
/// because both ends live in `vyre-libs`; the edge itself is the same one, and
/// the fix line that told a caller to hoist the shared piece into
/// `vyre-primitives` no longer names a place a composition can go.
///
/// A substrate directory is still walked as a source dialect, so an edge the
/// other way, from the substrate up into a dialect, is still flagged. That
/// direction is a layering inversion and never legal.
pub(super) const KERNEL_SUBSTRATE_DIRS: [&str; 18] = [
    "bitset",
    "decode",
    "fixpoint",
    "geom",
    "graph",
    "hash",
    "label",
    "matching",
    "math",
    "nfa",
    "nn",
    "opt",
    "parsing",
    "predicate",
    "reduce",
    "text",
    "topology",
    "visual",
];

/// Whether an edge whose target is the directory `name` is composition rather
/// than reach-through.
///
/// Only the target side of an edge is exempt. The substrate directory itself
/// stays in the dialect set `list_dialect_dirs` returns, so its own files are
/// still walked and an import from the substrate into a dialect is still a
/// finding.
pub(super) fn is_substrate_target(name: &str) -> bool {
    KERNEL_SUBSTRATE_DIRS.contains(&name)
}

/// Kernel-substrate rows whose directory under `libs_src` carries no Rust source.
pub(super) fn dead_substrate_rows(libs_src: &std::path::Path) -> Vec<&'static str> {
    KERNEL_SUBSTRATE_DIRS
        .into_iter()
        .filter(|dir| !structure_gate::source_scan::carries_rust_source(&libs_src.join(dir)))
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
    use super::*;

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

    /// WHY: the composition domains moved out of `vyre-primitives` into
    /// `vyre-libs/src`, so an edge a dialect always had, and always spelled
    /// `vyre_primitives::<domain>::…`, is now spelled `crate::<domain>::…`. The
    /// reach-through audit read the new spelling as a cross-dialect edge and
    /// produced 440 findings for edges nobody changed. The exemption is on the
    /// target side only, which is the whole contract: composing downward onto
    /// the substrate is legal, and importing upward out of the substrate into a
    /// dialect is a layering inversion that must stay a finding.
    ///
    /// The tempting wrong fix is to drop the substrate directories out of
    /// `list_dialect_dirs`. That silences the same 440 findings and also stops
    /// walking the substrate as a source, so the inversion goes unaudited. This
    /// test goes red on that fix.
    ///
    /// What this does not catch: a substrate row added for a directory that is
    /// really a dialect. That judgement is the reviewer's and the row carries
    /// it, exactly as a shared-plumbing row does.
    #[test]
    fn the_substrate_is_an_exempt_target_and_still_an_audited_source() {
        let libs_src = tempfile::tempdir().expect("temporary vyre-libs/src");
        let substrate = KERNEL_SUBSTRATE_DIRS[0];
        let dialect = "solvers";
        assert!(
            !KERNEL_SUBSTRATE_DIRS.contains(&dialect),
            "the control directory must not itself be substrate"
        );
        for dir in [substrate, dialect] {
            std::fs::create_dir(libs_src.path().join(dir)).expect("directory");
        }

        assert!(
            is_substrate_target(substrate),
            "a dialect composing onto the substrate is not reaching through"
        );
        assert!(
            !is_substrate_target(dialect),
            "a dialect naming another dialect is still reaching through"
        );

        let (dialects, errors) = list_dialect_dirs(libs_src.path());
        assert!(errors.is_empty(), "the temporary tree reads cleanly");
        let mut names = dialects
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect::<Vec<_>>();
        names.sort_unstable();
        let mut expected = vec![substrate, dialect];
        expected.sort_unstable();
        assert_eq!(
            names, expected,
            "the substrate is still walked as a source dialect, so an import out of it into a dialect is still audited"
        );
    }

    /// WHY: the kernel-substrate list is consumed by a directory filter, the
    /// same shape as the shared-plumbing list, so a row naming a directory that
    /// moved or was renamed suppresses nothing and reads as a reviewed
    /// exemption. Both directions are proved against a tree the check is handed.
    #[test]
    fn a_substrate_row_without_a_directory_behind_it_is_dead() {
        let libs_src = tempfile::tempdir().expect("temporary vyre-libs/src");

        assert_eq!(
            dead_substrate_rows(libs_src.path()),
            KERNEL_SUBSTRATE_DIRS.to_vec(),
            "every row is dead against a tree that holds none of them"
        );

        for dir in KERNEL_SUBSTRATE_DIRS {
            std::fs::create_dir(libs_src.path().join(dir)).expect("substrate directory");
        }
        assert_eq!(
            dead_substrate_rows(libs_src.path()),
            Vec::<&str>::new(),
            "no row is dead once every one of them names a directory"
        );

        let first = KERNEL_SUBSTRATE_DIRS[0];
        std::fs::remove_dir(libs_src.path().join(first)).expect("remove one substrate directory");
        assert_eq!(
            dead_substrate_rows(libs_src.path()),
            vec![first],
            "the row whose directory went away is the one reported"
        );
    }
}
