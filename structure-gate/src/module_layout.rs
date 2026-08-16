//! Where a module lives and what it is called.
//!
//! A reader looking for a concept opens the file whose name states it. Every
//! rule here judges that: a module file beside its own directory, a name that
//! states no contract, a number that distinguishes siblings, and a file that
//! repeats the directory it sits in. All of them read paths, so none of them
//! touches the filesystem.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// Trees a crate compiles source from, judged by the name rules.
///
/// `benches` and `examples` are in for the same reason `tests` is: a reader
/// looking for the fixture a benchmark uses reads its file name first.
pub(crate) const SOURCE_TREES: &[&str] = &["src", "tests", "benches", "examples"];

/// Names that state no contract.
///
/// A file called `helpers`, `types` or `utils` says nothing about what is
/// inside it, so finding the thing it holds means opening it, and deciding
/// where a new item goes means giving up and adding it there. Every name here
/// has that property; a name that states its contents does not.
///
/// The same word as a suffix is the same dumping ground with a qualifier
/// bolted on: `foo_ext` is whatever `foo` had no room for, and `spec_types` is
/// whatever the spec needed a home for. `is_banned_module_name` derives the
/// suffix family from this list so the two cannot drift apart.
const BANNED_MODULE_NAMES: &[&str] = &[
    "base", "common", "core", "ext", "extra", "glue", "helper", "helpers", "impl", "inner", "misc",
    "shared", "shim", "stuff", "support", "things", "types", "util", "utils", "wrapper",
];

/// Committed snapshot of the API each publishable crate reaches out with.
pub(crate) const PUBLIC_API_SNAPSHOT_DIR: &str = "docs/public-api";

/// One crate the module-layout rules judge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrateRoot {
    /// Checkout-relative crate directory.
    pub directory: String,
    /// Identifier a consumer writes to name this crate.
    ///
    /// Read from the manifest, `[lib] name` first and the package name second,
    /// never from the directory. A crate whose directory does not spell its
    /// package name would otherwise be looked up in the public-API snapshot
    /// under a name the snapshot never uses, and every published module of that
    /// crate would lose the exemption that keeps a published name stable.
    pub ident: String,
}

/// Reject a module file that sits beside a directory of the same name.
///
/// `foo.rs` next to `foo/` splits one module across two places in a listing:
/// the file declaring the module's own items sorts away from the directory
/// holding its children, so a reader looking for `foo` opens whichever the
/// editor shows first and finds half of it. `foo/mod.rs` is the same module
/// with the file and its children in one place.
///
/// The rule reads `src/` only. An integration test binary is named by its own
/// file, so `tests/foo.rs` beside `tests/foo/` is a binary next to its
/// fixtures rather than one module in two places.
#[must_use]
pub fn sibling_module_failures(module_files: &[String]) -> Vec<String> {
    let mut directories: BTreeSet<&str> = BTreeSet::new();
    for file in module_files {
        let mut rest = file.as_str();
        while let Some((parent, _)) = rest.rsplit_once('/') {
            if !directories.insert(parent) {
                break;
            }
            rest = parent;
        }
    }
    let mut failures: Vec<String> = module_files
        .iter()
        .filter_map(|file| {
            let stem = file.strip_suffix(".rs")?;
            directories.contains(stem).then(|| {
                format!(
                    "`{file}` sits beside its own directory `{stem}/`; one module is one place, so it belongs at `{stem}/mod.rs`"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// Reject a file, module or binary whose name states no contract.
///
/// Judged over every source tree a crate compiles, not `src/` alone. The
/// prohibition was written for modules and went unenforced against
/// test-adjacent files, which is where the population moved: at the last count
/// 15 of 16 remaining banned names were `tests/common/mod.rs` or
/// `tests/support/mod.rs`.
///
/// A module is exempt only while the committed public-API snapshot publishes
/// it: renaming a published module renames a path a consumer already imports,
/// and this gate is not what decides to break one. The exemption is read from
/// the snapshot at run time, so it lapses by itself once the module stops
/// being published, and a crate with no snapshot cannot claim it at all. A file
/// outside `src/` has no public path and no exemption.
///
/// A Cargo binary root is judged by the binary's name, which is the word a
/// reader types to run it: an executable called `utils` states no more than a
/// module of that name. A binary has no module path, so no snapshot exempts one.
///
/// A name ending in two digit runs, `validation_findings_12_20`, names the
/// ticket that produced the file rather than the contract inside it, and the
/// ticket is closed by the time anyone reads the name.
///
/// What this does not catch: a specific name that is still wrong for its
/// contents, a published module that carries a banned name, and a directory
/// with no `mod.rs`, whose name no file states. The second one shows up as a
/// snapshot diff in the change that publishes it.
#[must_use]
pub fn generic_module_name_failures(
    module_files: &[String],
    crate_roots: &[CrateRoot],
    published_modules: &[String],
) -> Vec<String> {
    let published: BTreeSet<&str> = published_modules.iter().map(String::as_str).collect();
    let mut failures = Vec::new();
    for file in module_files {
        if let Some(binary) = binary_name_of(file) {
            if is_banned_module_name(binary) {
                failures.push(format!(
                    "`{file}` names the binary `{binary}`, which states no contract; name it for what the binary does"
                ));
            }
            continue;
        }
        let Some(name) = judged_name_of(file) else {
            continue;
        };
        if let Some(range) = ticket_range_of(name) {
            failures.push(format!(
                "`{file}` is named for ticket range `{range}`, not for a contract; name it for what it holds"
            ));
            continue;
        }
        if !is_banned_module_name(name) {
            continue;
        }
        let path = module_path_of(file, crate_roots);
        if path.as_deref().is_some_and(|path| published.contains(path)) {
            continue;
        }
        let published_note = path.map_or_else(String::new, |path| {
            format!(" ({path} is published at no public path, so renaming it breaks nothing)")
        });
        failures.push(format!(
            "`{file}` declares module `{name}`, which states no contract; name it for what it holds{published_note}"
        ));
    }
    failures.sort();
    failures.dedup();
    failures
}

/// Reject sibling files distinguished only by a number.
///
/// `nodes_00.rs` through `nodes_09.rs` in one directory convey nothing about
/// which of the ten classifies a given node, so finding the one that answers a
/// question means opening all ten, and a new case goes into whichever file the
/// author had open. A number inside a name that means something, `crc32`,
/// `float16`, `flash_attention_2`, has no numbered sibling, which is what
/// separates the two: the defect is the number carrying the distinction.
#[must_use]
pub fn numbered_sibling_failures(source_files: &[String]) -> Vec<String> {
    let mut families: BTreeMap<(&str, &str), Vec<&String>> = BTreeMap::new();
    for file in source_files {
        let Some(name) = judged_name_of(file) else {
            continue;
        };
        let Some(stem) = numbered_stem_of(name) else {
            continue;
        };
        let directory = file.rsplit_once('/').map_or("", |(head, _)| head);
        families.entry((directory, stem)).or_default().push(file);
    }
    let mut failures: Vec<String> = families
        .into_iter()
        .filter(|(_, family)| family.len() > 1)
        .flat_map(|((_, stem), family)| {
            let count = family.len();
            family.into_iter().map(move |file| {
                format!(
                    "`{file}` is one of {count} `{stem}_N` siblings, so the number carries the \
                     distinction and the name carries none; name each for what it holds"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// Reject a file that repeats the name of the directory holding it.
///
/// `hardware/fma_f32/fma_f32.rs` states its contents once and its location
/// twice, and the reader who opens the directory has to decide whether the file
/// is the module or a part of it. The module is the directory, so the file is
/// `mod.rs`.
#[must_use]
pub fn directory_stutter_failures(source_files: &[String]) -> Vec<String> {
    let mut failures: Vec<String> = source_files
        .iter()
        .filter_map(|file| {
            let (parents, name) = file.rsplit_once('/')?;
            let stem = name.strip_suffix(".rs")?;
            let directory = parents.rsplit('/').next()?;
            (stem == directory).then(|| {
                format!(
                    "`{file}` repeats the directory `{directory}/` that holds it; the module is \
                     the directory, so this file is `{parents}/mod.rs`"
                )
            })
        })
        .collect();
    failures.sort();
    failures.dedup();
    failures
}

/// The stem a numbered sibling shares with its family, or `None`.
///
/// `nodes_09` is `nodes`; `float16` and `sha256` have no `_` before the digits,
/// so the digits are part of one word rather than a sibling index.
fn numbered_stem_of(name: &str) -> Option<&str> {
    let (stem, digits) = name.rsplit_once('_')?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) && !stem.is_empty())
        .then_some(stem)
}

/// The `<digits>_<digits>` tail of a name written for a ticket, or `None`.
fn ticket_range_of(name: &str) -> Option<String> {
    let (head, second) = name.rsplit_once('_')?;
    let (_, first) = head.rsplit_once('_')?;
    (!first.is_empty()
        && !second.is_empty()
        && first.bytes().all(|byte| byte.is_ascii_digit())
        && second.bytes().all(|byte| byte.is_ascii_digit()))
    .then(|| format!("{first}_{second}"))
}

/// True when a name is a dumping ground by name alone.
///
/// A banned word standing alone, or the same word as a `_` suffix.
fn is_banned_module_name(name: &str) -> bool {
    BANNED_MODULE_NAMES.contains(&name)
        || BANNED_MODULE_NAMES.iter().any(|banned| {
            name.len() > banned.len() + 1
                && name.ends_with(banned)
                && name.as_bytes()[name.len() - banned.len() - 1] == b'_'
        })
}

/// The name a source file states, or `None` for a crate or binary root.
///
/// A `mod.rs` is named by its directory, which is the whole point of the
/// layout: reading the file name alone would judge every module in the
/// workspace as being called `mod`. A file under `tests/`, `benches/` or
/// `examples/` is judged the same way, because a reader looking for a fixture
/// reads that name for the same reason.
fn judged_name_of(file: &str) -> Option<&str> {
    if binary_name_of(file).is_some() {
        return None;
    }
    let inside = SOURCE_TREES
        .iter()
        .filter_map(|tree| file.split_once(&format!("/{tree}/")).map(|(_, rest)| rest))
        .min_by_key(|rest| rest.len())?;
    match inside.rsplit('/').next()? {
        "lib.rs" | "main.rs" => None,
        "mod.rs" => inside.rsplit('/').nth(1),
        other => other.strip_suffix(".rs"),
    }
}

/// The binary name a `src/` file declares, or `None` when it is not one.
///
/// Cargo takes both `src/bin/<name>.rs` and `src/bin/<name>/main.rs` as binary
/// roots. Anything deeper under `src/bin/<name>/` is an ordinary module of that
/// binary and is judged as one.
fn binary_name_of(file: &str) -> Option<&str> {
    let (_, inside) = file.split_once("/src/")?;
    let after = inside.strip_prefix("bin/")?;
    match after.split_once('/') {
        None => after.strip_suffix(".rs"),
        Some((name, "main.rs")) => Some(name),
        Some(_) => None,
    }
}

/// The module path a `src/` file declares, as a consumer writes it.
///
/// `vyre-libs/src/parsing/core/mod.rs` is `vyre_libs::parsing::core`. The crate
/// part comes from the [`CrateRoot`] whose directory holds the file, so it is
/// the name the manifest declares rather than the name the directory spells.
/// `None` when no scanned crate holds the file, which is the only honest answer:
/// a guessed crate name would be looked up in the public-API snapshot and miss.
fn module_path_of(file: &str, crate_roots: &[CrateRoot]) -> Option<String> {
    let mut path = crate_roots
        .iter()
        .filter(|crate_root| file.starts_with(&format!("{}/src/", crate_root.directory)))
        .max_by_key(|crate_root| crate_root.directory.len())
        .map(|crate_root| crate_root.ident.clone())?;
    let (_, inside) = file.split_once("/src/")?;
    let name = inside.rsplit('/').next()?;
    let parents = inside.rsplit_once('/').map_or("", |(head, _)| head);
    let tail = if name == "mod.rs" {
        Cow::Borrowed(parents)
    } else {
        let stem = name.strip_suffix(".rs")?;
        if parents.is_empty() {
            Cow::Borrowed(stem)
        } else {
            Cow::Owned(format!("{parents}/{stem}"))
        }
    };
    for segment in tail.split('/').filter(|segment| !segment.is_empty()) {
        path.push_str("::");
        path.push_str(segment);
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(files: &[&str]) -> Vec<String> {
        files.iter().map(|file| (*file).to_string()).collect()
    }

    #[test]
    fn a_module_file_beside_its_own_directory_is_rejected() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-libs/src/rule/admission.rs",
        ]));

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("vyre-libs/src/rule/mod.rs"),
            "{failures:?}"
        );
    }

    #[test]
    fn a_module_file_is_judged_against_a_directory_holding_no_direct_source() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-libs/src/rule/admission/window.rs",
        ]));

        assert_eq!(failures.len(), 1, "{failures:?}");
    }

    #[test]
    fn a_module_inside_its_own_directory_is_accepted() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule/mod.rs",
            "vyre-libs/src/rule/admission.rs",
            "vyre-libs/src/lib.rs",
        ]));

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_same_named_file_in_another_directory_is_not_a_pair() {
        let failures = sibling_module_failures(&paths(&[
            "vyre-libs/src/rule.rs",
            "vyre-driver/src/rule/admission.rs",
        ]));

        assert!(failures.is_empty(), "{failures:?}");
    }

    fn crate_roots(pairs: &[(&str, &str)]) -> Vec<CrateRoot> {
        pairs
            .iter()
            .map(|(directory, ident)| CrateRoot {
                directory: (*directory).to_string(),
                ident: (*ident).to_string(),
            })
            .collect()
    }

    #[test]
    fn every_banned_module_name_is_rejected_as_a_file_and_as_a_directory() {
        for name in BANNED_MODULE_NAMES {
            let flat = format!("vyre-libs/src/scan/{name}.rs");
            let nested = format!("vyre-libs/src/scan/{name}/mod.rs");
            let failures = generic_module_name_failures(
                &[flat, nested],
                &crate_roots(&[("vyre-libs", "vyre_libs")]),
                &[],
            );

            assert_eq!(failures.len(), 2, "{name}: {failures:?}");
            assert!(
                failures
                    .iter()
                    .all(|failure| failure.contains(&format!("vyre_libs::scan::{name}"))),
                "{name}: {failures:?}"
            );
        }
    }

    #[test]
    fn a_qualifier_suffix_is_rejected_as_a_file_and_as_a_directory() {
        let failures = generic_module_name_failures(
            &paths(&[
                "vyre-libs/src/scan/window_ext.rs",
                "vyre-libs/src/scan/region_ext/mod.rs",
                "xtask/src/bin/dump_ext.rs",
            ]),
            &crate_roots(&[("vyre-libs", "vyre_libs"), ("xtask", "xtask")]),
            &[],
        );

        assert_eq!(failures.len(), 3, "{failures:?}");
    }

    #[test]
    fn a_published_module_keeps_its_name() {
        let files = paths(&["vyre-libs/src/parsing/core/mod.rs"]);
        let roots = crate_roots(&[("vyre-libs", "vyre_libs")]);

        assert!(generic_module_name_failures(
            &files,
            &roots,
            &["vyre_libs::parsing::core".to_string()]
        )
        .is_empty());
        assert_eq!(
            generic_module_name_failures(&files, &roots, &["vyre_libs::parsing".to_string()]).len(),
            1
        );
    }

    #[test]
    fn the_exemption_is_keyed_on_the_name_the_manifest_declares() {
        let files = paths(&["fuzz/src/harness/types/mod.rs"]);
        let published = ["vyre_fuzz::harness::types".to_string()];

        assert!(
            generic_module_name_failures(
                &files,
                &crate_roots(&[("fuzz", "vyre_fuzz")]),
                &published
            )
            .is_empty(),
            "a published module lost its exemption because the crate was named after its directory"
        );
        assert_eq!(
            generic_module_name_failures(&files, &crate_roots(&[("fuzz", "fuzz")]), &published)
                .len(),
            1
        );
    }

    #[test]
    fn a_module_in_no_scanned_crate_is_still_reported() {
        let failures = generic_module_name_failures(&paths(&["stray/src/types/mod.rs"]), &[], &[]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            !failures[0].contains("published at no public path"),
            "the message claimed a public path it could not resolve: {failures:?}"
        );
    }

    #[test]
    fn a_crate_root_carries_no_module_name() {
        assert_eq!(judged_name_of("vyre-libs/src/lib.rs"), None);
        assert_eq!(judged_name_of("conform/vyre-conform/src/main.rs"), None);
        assert_eq!(judged_name_of("vyre-libs/src/scan/mod.rs"), Some("scan"));
        assert_eq!(
            judged_name_of("vyre-libs/src/scan/window.rs"),
            Some("window")
        );
    }

    #[test]
    fn every_source_tree_is_judged_and_a_file_in_none_is_not() {
        for tree in SOURCE_TREES {
            assert_eq!(
                judged_name_of(&format!("vyre-libs/{tree}/parity/mod.rs")),
                Some("parity"),
                "the {tree} tree went unjudged"
            );
            assert_eq!(
                judged_name_of(&format!("vyre-libs/{tree}/parity_support.rs")),
                Some("parity_support")
            );
        }
        assert_eq!(judged_name_of("release/changes/support.rs"), None);
        assert_eq!(
            judged_name_of("vyre-libs/tests/support/nested/deep/util.rs"),
            Some("util"),
            "a file nested under a judged tree is judged by its own name"
        );
    }

    #[test]
    fn a_number_is_a_defect_only_when_it_distinguishes_siblings() {
        let family: Vec<String> = (0..3)
            .map(|index| format!("vyre-libs/src/classify/nodes_0{index}.rs"))
            .collect();
        let failures = numbered_sibling_failures(&family);
        assert_eq!(failures.len(), 3, "{failures:?}");
        assert!(failures[0].contains("3 `nodes_N` siblings"), "{failures:?}");

        for lone in [
            "vyre-libs/src/nn/attention/flash_attention_2.rs",
            "vyre-primitives/src/hash/crc32.rs",
            "vyre-primitives/src/math/float16.rs",
            "vyre-libs/src/classify/nodes_00.rs",
        ] {
            assert!(
                numbered_sibling_failures(&paths(&[lone])).is_empty(),
                "{lone} has no numbered sibling, so its digits carry meaning"
            );
        }
        assert!(
            numbered_sibling_failures(&paths(&[
                "vyre-libs/src/classify/nodes_00.rs",
                "vyre-libs/src/emit/nodes_01.rs",
            ]))
            .is_empty(),
            "siblings are per directory; two directories are not one family"
        );
    }

    #[test]
    fn a_ticket_range_is_rejected_and_a_single_number_is_not() {
        let failures = generic_module_name_failures(
            &paths(&["vyre-foundation/tests/validation_findings_12_20.rs"]),
            &crate_roots(&[("vyre-foundation", "vyre_foundation")]),
            &[],
        );
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("ticket range `12_20`"), "{failures:?}");
        assert!(
            generic_module_name_failures(
                &paths(&["xtask-evidence/src/semantic/proof_workloads_12.rs"]),
                &crate_roots(&[("xtask-evidence", "xtask_evidence")]),
                &[],
            )
            .is_empty(),
            "one number can be a count or a size; a range names a ticket"
        );
    }

    #[test]
    fn a_file_repeating_its_directory_is_rejected() {
        let failures = directory_stutter_failures(&paths(&[
            "vyre-intrinsics/src/hardware/fma_f32/fma_f32.rs",
            "vyre-intrinsics/src/hardware/fma_f32/mod.rs",
            "vyre-intrinsics/src/hardware/fma_f32/lowering.rs",
        ]));
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("fma_f32/mod.rs`"),
            "the failure must name where the file belongs: {failures:?}"
        );
    }

    #[test]
    fn a_binary_root_is_judged_by_the_binary_name() {
        for name in BANNED_MODULE_NAMES {
            let flat = format!("xtask/src/bin/{name}.rs");
            let nested = format!("xtask/src/bin/{name}/main.rs");
            let failures = generic_module_name_failures(
                &[flat, nested],
                &crate_roots(&[("xtask", "xtask")]),
                &["xtask::bin".to_string()],
            );

            assert_eq!(failures.len(), 2, "{name}: {failures:?}");
            assert!(
                failures
                    .iter()
                    .all(|failure| failure.contains(&format!("names the binary `{name}`"))),
                "{name}: {failures:?}"
            );
        }
    }

    #[test]
    fn a_named_binary_and_its_own_modules_are_told_apart() {
        assert_eq!(
            binary_name_of("xtask/src/bin/scaffold_rule.rs"),
            Some("scaffold_rule")
        );
        assert_eq!(
            binary_name_of("xtask-registry/src/bin/vyre_new_op/main.rs"),
            Some("vyre_new_op")
        );
        assert_eq!(
            binary_name_of("xtask-registry/src/bin/vyre_new_op/run.rs"),
            None
        );
        assert_eq!(binary_name_of("vyre-libs/src/scan/window.rs"), None);
        assert_eq!(judged_name_of("xtask/src/bin/scaffold_rule.rs"), None);
        assert_eq!(
            judged_name_of("xtask-registry/src/bin/vyre_new_op/helpers.rs"),
            Some("helpers")
        );
    }

    #[test]
    fn a_descriptively_named_binary_is_accepted() {
        let failures = generic_module_name_failures(
            &paths(&[
                "xtask/src/bin/scaffold_rule.rs",
                "xtask-registry/src/bin/vyre_new_op/main.rs",
            ]),
            &crate_roots(&[("xtask", "xtask"), ("xtask-registry", "xtask_registry")]),
            &[],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_module_path_is_read_through_the_src_boundary() {
        let roots = crate_roots(&[
            ("conform/vyre-conform", "vyre_conform"),
            ("vyre-libs", "vyre_libs"),
        ]);

        assert_eq!(
            module_path_of("conform/vyre-conform/src/report/common/mod.rs", &roots).as_deref(),
            Some("vyre_conform::report::common")
        );
        assert_eq!(
            module_path_of("vyre-libs/src/types.rs", &roots).as_deref(),
            Some("vyre_libs::types")
        );
        assert_eq!(module_path_of("vyre-libs/src/types.rs", &[]), None);
    }

    #[test]
    fn a_descriptive_module_name_is_accepted() {
        let failures = generic_module_name_failures(
            &paths(&[
                "vyre-libs/src/scan/regex_dfa.rs",
                "vyre-libs/src/graph/dispatch/mod.rs",
                "vyre-libs/src/lib.rs",
            ]),
            &crate_roots(&[("vyre-libs", "vyre_libs")]),
            &[],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }
}
