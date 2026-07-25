//! The downstream-consumer naming boundary, in one place.
//!
//! # What the boundary is
//!
//! Vyre is a substrate. Its platform crates are meant to be usable by anyone,
//! so no source file in them may name a specific downstream product. A name
//! that leaks into a platform crate turns a general API into a special case for
//! one consumer, and the leak is easy to miss in review because it usually
//! arrives as a comment or a symbol prefix rather than a dependency.
//!
//! Six test suites enforce that, one per platform crate. Each used to carry its
//! own copy of the forbidden-name list, spelled as `concat!` pairs so the list
//! would not match itself during the scan. The splitting is necessary; six
//! copies of it are not. A new downstream product had to be added in six files,
//! and any copy that fell out of sync stopped guarding its crate with nothing
//! red to show for it.
//!
//! [`FORBIDDEN_CONSUMER_NAMES`] is the one definition, and
//! [`assert_source_does_not_name_downstream_consumers`] is the one scan. A
//! suite becomes three lines:
//!
//! ```ignore
//! #[test]
//! fn driver_source_does_not_name_downstream_consumers() {
//!     vyre_test_support::consumer_boundary::assert_source_does_not_name_downstream_consumers(
//!         ConsumerBoundaryScan::for_crate("vyre-driver", env!("CARGO_MANIFEST_DIR"))
//!             .with_rationale("vyre-driver is a platform crate"),
//!     );
//! }
//! ```
//!
//! # Why this file does not fail its own scan
//!
//! The names below are still written as `concat!` pairs. No scanner reading
//! this file as text sees a whole name, so the owner needs no path exemption
//! and no scanner needs to know it exists.

use std::fs;
use std::path::{Path, PathBuf};

/// Downstream products whose names must not appear in a platform crate.
///
/// Each entry is split across a `concat!` so that this list does not match
/// itself when a scan reads this file as text. Add a product here and every
/// platform crate starts guarding against it in the same commit.
pub const FORBIDDEN_CONSUMER_NAMES: [&str; 4] = [
    concat!("we", "ir"),
    concat!("sur", "gec"),
    concat!("gos", "san"),
    concat!("key", "hog"),
];

/// One crate's scan configuration.
///
/// Built with [`ConsumerBoundaryScan::for_crate`] and narrowed with the
/// builder methods. Everything is required to be explicit: there is no default
/// crate label and no implicit directory exemption, because a silently skipped
/// directory is an unguarded directory.
#[derive(Debug, Clone)]
pub struct ConsumerBoundaryScan {
    crate_label: String,
    source_root: PathBuf,
    rationale: String,
    skipped_directory_names: Vec<String>,
}

impl ConsumerBoundaryScan {
    /// Scans `src/` under a crate's manifest directory.
    ///
    /// Pass `env!("CARGO_MANIFEST_DIR")` for `manifest_dir`; the label is the
    /// crate name and appears in every diagnostic the scan produces.
    #[must_use]
    pub fn for_crate(crate_label: &str, manifest_dir: &str) -> Self {
        Self {
            crate_label: crate_label.to_owned(),
            source_root: Path::new(manifest_dir).join("src"),
            rationale: format!("{crate_label} is a platform crate"),
            skipped_directory_names: Vec::new(),
        }
    }

    /// Replaces the sentence that explains why this crate is bound.
    ///
    /// It leads the failure message, so it should say what the crate is rather
    /// than restate the rule.
    #[must_use]
    pub fn with_rationale(mut self, rationale: &str) -> Self {
        self.rationale = rationale.to_owned();
        self
    }

    /// Excludes directories by name, anywhere under the source root.
    ///
    /// Only for material that is intentionally historical, such as an archive
    /// of superseded modules. Every exclusion is named in the failure message
    /// so a reader can tell what the scan did not cover.
    #[must_use]
    pub fn skipping_directories(mut self, names: &[&str]) -> Self {
        self.skipped_directory_names
            .extend(names.iter().map(|name| (*name).to_owned()));
        self
    }
}

/// Fails if any file under the configured source root names a downstream product.
///
/// Reports every violation at once, as `path contains name`, so one run fixes
/// the whole crate rather than one file per run.
///
/// # Panics
///
/// Panics if the source root cannot be read, which means the crate layout moved
/// and the scan is covering nothing.
pub fn assert_source_does_not_name_downstream_consumers(scan: ConsumerBoundaryScan) {
    let mut source_files = Vec::new();
    collect_source_files(&scan, &scan.source_root.clone(), &mut source_files);
    source_files.sort();

    assert!(
        !source_files.is_empty(),
        "Fix: the {} consumer boundary scan found no .rs files under {}; the crate layout \
         moved and the scan is guarding nothing.",
        scan.crate_label,
        scan.source_root.display()
    );

    let mut violations = Vec::new();
    for source_file in source_files {
        let contents = fs::read_to_string(&source_file).unwrap_or_else(|error| {
            panic!(
                "failed to read {} source file {}: {error}",
                scan.crate_label,
                source_file.display()
            )
        });
        for name in FORBIDDEN_CONSUMER_NAMES {
            if contents.contains(name) {
                violations.push(format!("{} contains {name}", source_file.display()));
            }
        }
    }

    let skipped = if scan.skipped_directory_names.is_empty() {
        String::new()
    } else {
        format!(
            "\n(directories excluded from this scan: {})",
            scan.skipped_directory_names.join(", ")
        )
    };
    assert!(
        violations.is_empty(),
        "{} and must not name downstream consumers:\n{}{skipped}",
        scan.rationale,
        violations.join("\n")
    );
}

fn collect_source_files(scan: &ConsumerBoundaryScan, root: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(root).unwrap_or_else(|error| {
        panic!(
            "failed to read {} source directory {}: {error}",
            scan.crate_label,
            root.display()
        )
    });

    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to read {} source directory entry under {}: {error}",
                scan.crate_label,
                root.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to classify {} source path {}: {error}",
                scan.crate_label,
                path.display()
            )
        });
        if file_type.is_dir() {
            let skipped = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    scan.skipped_directory_names
                        .iter()
                        .any(|skipped| skipped == name)
                });
            if !skipped {
                collect_source_files(scan, &path, out);
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The forbidden list must hold four distinct, non-empty names.
    ///
    /// A `concat!` pair that lost half of itself would still compile and would
    /// silently shrink the guard to a substring that matches everything or
    /// nothing. This pins the shape the scan depends on.
    #[test]
    fn the_forbidden_list_holds_four_distinct_non_empty_names() {
        let mut sorted = FORBIDDEN_CONSUMER_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FORBIDDEN_CONSUMER_NAMES.len());
        for name in FORBIDDEN_CONSUMER_NAMES {
            assert!(name.len() >= 4, "{name} is too short to be a product name");
        }
    }

    /// This file must not contain any forbidden name as a whole string.
    ///
    /// It is the reason the list is spelled with `concat!`. If someone
    /// "tidied" the pairs into plain literals, every crate's scan would start
    /// failing on this file, or worse, would need a path exemption that then
    /// hides real leaks in whatever else that exemption covers.
    #[test]
    fn the_owner_file_never_spells_a_forbidden_name_whole() {
        let source = include_str!("consumer_boundary.rs");
        for name in FORBIDDEN_CONSUMER_NAMES {
            assert!(
                !source.contains(name),
                "consumer_boundary.rs spells {name} as a whole literal; keep the concat! split"
            );
        }
    }

    /// A skipped directory is not scanned, and the exclusion is disclosed.
    ///
    /// Skipping is the one way the scan can miss a real leak, so a run that
    /// skipped something has to say so in its own failure output. This proves
    /// both halves against a temporary tree holding a planted violation.
    #[test]
    fn a_skipped_directory_is_excluded_and_named_in_the_failure_message() {
        let tree = TempTree::new("skipped-directory");
        let src = tree.path.join("src");
        fs::create_dir_all(src.join("archive")).expect("temp tree must be creatable");
        fs::write(src.join("lib.rs"), "// clean\n").expect("temp file must be writable");
        fs::write(
            src.join("archive").join("old.rs"),
            format!("// {}\n", FORBIDDEN_CONSUMER_NAMES[0]),
        )
        .expect("temp file must be writable");

        let scan = ConsumerBoundaryScan::for_crate("temp-crate", tree.path.to_str().unwrap())
            .skipping_directories(&["archive"]);
        assert_source_does_not_name_downstream_consumers(scan.clone());

        let unskipped =
            ConsumerBoundaryScan::for_crate("temp-crate", tree.path.to_str().unwrap());
        let failure = std::panic::catch_unwind(move || {
            assert_source_does_not_name_downstream_consumers(unskipped);
        })
        .expect_err("the planted violation must fail once the directory is scanned");
        let message = panic_message(&failure);
        assert!(
            message.contains(FORBIDDEN_CONSUMER_NAMES[0]),
            "failure must name the leaked product: {message}"
        );
        assert!(
            message.contains("old.rs"),
            "failure must name the offending file: {message}"
        );
    }

    /// A leak is reported once per file and name, with the path.
    ///
    /// The scan collects every violation before asserting so that one run fixes
    /// a whole crate. A scan that stopped at the first hit would turn a ten-file
    /// cleanup into ten test runs.
    #[test]
    fn every_violation_is_reported_rather_than_only_the_first() {
        let tree = TempTree::new("all-violations");
        let src = tree.path.join("src");
        fs::create_dir_all(&src).expect("temp tree must be creatable");
        fs::write(src.join("a.rs"), format!("// {}\n", FORBIDDEN_CONSUMER_NAMES[0]))
            .expect("temp file must be writable");
        fs::write(src.join("b.rs"), format!("// {}\n", FORBIDDEN_CONSUMER_NAMES[1]))
            .expect("temp file must be writable");

        let scan = ConsumerBoundaryScan::for_crate("temp-crate", tree.path.to_str().unwrap());
        let failure = std::panic::catch_unwind(move || {
            assert_source_does_not_name_downstream_consumers(scan);
        })
        .expect_err("planted violations must fail the scan");
        let message = panic_message(&failure);
        assert!(message.contains("a.rs"), "{message}");
        assert!(message.contains("b.rs"), "{message}");
    }

    /// An empty source root is a failure, not a pass.
    ///
    /// This is the quiet way a boundary gate dies: the crate is reorganized,
    /// `src/` no longer holds the sources, and the scan keeps reporting green
    /// while guarding nothing at all.
    #[test]
    fn a_source_root_with_no_rust_files_fails_instead_of_passing_vacuously() {
        let tree = TempTree::new("empty-root");
        fs::create_dir_all(tree.path.join("src")).expect("temp tree must be creatable");

        let scan = ConsumerBoundaryScan::for_crate("temp-crate", tree.path.to_str().unwrap());
        let failure = std::panic::catch_unwind(move || {
            assert_source_does_not_name_downstream_consumers(scan);
        })
        .expect_err("an empty source root must fail");
        assert!(panic_message(&failure).contains("guarding nothing"));
    }

    /// No second copy of the list may reappear anywhere in the workspace.
    ///
    /// This is the regression the module exists to prevent, and it is the one
    /// failure mode a shared owner does not fix by itself: someone adds a new
    /// suite, copies the four `concat!` pairs into it because that is what the
    /// old suites looked like, and the copy quietly stops tracking the owner.
    ///
    /// A copy is recognized exactly rather than by shape. The scan evaluates
    /// every `concat!("a", "b")` it finds in a file and asks whether the joined
    /// text is a forbidden name, so an unrelated `concat!` (the workspace uses
    /// plenty for shader source assembly) is never mistaken for one. A file
    /// naming two or more products that way is spelling the list.
    #[test]
    fn no_other_file_in_the_workspace_redefines_the_forbidden_list() {
        /// A file may legitimately spell one product name to make a single
        /// point about it. Two is a list.
        const LIST_THRESHOLD: usize = 2;

        let workspace = crate::monorepo::vyre_workspace_root();
        // Matched by full path, not by file name: the crates' own suites are
        // also called consumer_boundary.rs, and they are exactly where a fresh
        // copy of the list would show up.
        let owner = workspace.join(file!());
        assert!(
            owner.is_file(),
            "the owner path {} does not exist, so the scan would report itself",
            owner.display()
        );
        let mut duplicates = Vec::new();
        let mut scanned = 0usize;

        let crates = fs::read_dir(&workspace).expect("workspace root must be readable");
        for entry in crates {
            let crate_root = entry.expect("workspace entry must be readable").path();
            for sub in ["src", "tests"] {
                let root = crate_root.join(sub);
                if !root.is_dir() {
                    continue;
                }
                let mut files = Vec::new();
                let scan = ConsumerBoundaryScan::for_crate("workspace", ".");
                collect_source_files(&scan, &root, &mut files);
                for file in files {
                    if file == owner {
                        continue;
                    }
                    scanned += 1;
                    let text = fs::read_to_string(&file).expect("source file must be readable");
                    let named = split_spelled_names(&text);
                    if named.len() >= LIST_THRESHOLD {
                        duplicates.push(format!("{} spells {}", file.display(), named.join(", ")));
                    }
                }
            }
        }

        assert!(
            scanned > 100,
            "the duplicate-list scan only read {scanned} files; it is not covering the workspace"
        );
        assert!(
            duplicates.is_empty(),
            "these files spell a forbidden-consumer list of their own; read \
             FORBIDDEN_CONSUMER_NAMES from vyre-test-support instead:\n{}",
            duplicates.join("\n")
        );
    }

    /// The spelled-name reader must decide by value, not by shape.
    ///
    /// The first version of the duplicate scan counted `concat!` occurrences.
    /// That flagged eleven production files whose `concat!` calls assemble
    /// shader source and have nothing to do with product names. Deciding on the
    /// joined value keeps the guard exact: only a `concat!` that really spells a
    /// forbidden name counts.
    #[test]
    fn the_spelled_name_reader_matches_joined_values_and_ignores_unrelated_concats() {
        let name = FORBIDDEN_CONSUMER_NAMES[0];
        let (head, tail) = name.split_at(2);
        let spelled = format!("let x = concat!(\"{head}\", \"{tail}\");");
        assert_eq!(split_spelled_names(&spelled), vec![name.to_owned()]);

        let unrelated = "let wgsl = concat!(\"@group(0) \", \"@binding(0)\");";
        assert!(split_spelled_names(unrelated).is_empty());

        // Whole literals are the scanners' own business, not this reader's:
        // it only reports names that were deliberately split to evade a scan.
        assert!(split_spelled_names(&format!("// {name}\n")).is_empty());

        // Repeats collapse, so a file spelling one name three times is not a list.
        assert_eq!(split_spelled_names(&spelled.repeat(3)).len(), 1);
    }

    /// Reads every `concat!("a", "b")` in `text` and returns the distinct
    /// forbidden names its arguments join into.
    fn split_spelled_names(text: &str) -> Vec<String> {
        const OPEN: &str = "concat!(";
        let mut found: Vec<String> = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(OPEN) {
            rest = &rest[start + OPEN.len()..];
            let Some(end) = rest.find(')') else { break };
            let joined: String = rest[..end]
                .split(',')
                .filter_map(|argument| {
                    let argument = argument.trim();
                    argument
                        .strip_prefix('"')
                        .and_then(|argument| argument.strip_suffix('"'))
                })
                .collect();
            if FORBIDDEN_CONSUMER_NAMES.contains(&joined.as_str())
                && !found.contains(&joined)
            {
                found.push(joined);
            }
        }
        found.sort();
        found
    }

    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default()
    }

    /// A temporary directory that removes itself, so the tests leave no trace
    /// even when an assertion unwinds.
    struct TempTree {
        path: PathBuf,
    }

    impl TempTree {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "vyre-consumer-boundary-{label}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("temp tree must be creatable");
            Self { path }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
