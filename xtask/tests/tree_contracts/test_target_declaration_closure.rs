//! Every declared test target names a file, and every test file a target compiles.
//!
//! WHY: a member that sets `autotests = false` gets no test target from cargo's
//! discovery, so the manifest is the only thing that decides what runs. Both
//! ends of that declaration fail silently.
//!
//! A file added directly under `tests/` that no `[[test]]` row names and no
//! harness declares as a module compiles nowhere. The suite stays green and the
//! assertions in the file are gone, while the file's presence reads as coverage.
//! `test-target-membership` owns that rule across the whole `tests/` subtree and
//! subtracts two classes from it, material a `src/` module includes and fixture
//! data a compiler harness reads by glob. Neither class may sit directly under
//! `tests/`, so at that one depth the rule holds with nothing subtracted, and a
//! failure names the file rather than a set the exclusions had to be read
//! against. Both read the same `ownership` walk, so there is one answer to which
//! target compiles which file.
//!
//! A `[[test]]` row whose path names no file is the same failure from the other
//! side. Cargo rejects a missing target root only when it builds that target,
//! and a row carrying `required-features` is skipped whenever those features are
//! off, so the row can name a deleted file for as long as nobody runs that
//! feature set. `test-target-membership` skips such a row outright, which is
//! what leaves this direction unheld.
//!
//! Both answers are derived from the checkout at run time: members come from
//! `workspace.members`, rows from each member's own manifest, and files from the
//! tree walk. Adding a member, a row, or a file changes what this judges without
//! anyone editing it.
//!
//! # What it does not judge
//!
//! Only files directly under `tests/` are required to be compiled. A
//! subdirectory holds fixture data a harness reads by glob and unit-test
//! material a `src/` module includes through `#[path]`; `test-material-placement`
//! owns where the second may live and neither is a test target's module.

use xtask::gates::scan::{Member, Tree};
use xtask::gates::test_target_membership::ownership;

use super::workspace_sources::workspace_root;

/// The checkout this run stands in, listed once per test.
fn tree() -> Tree {
    Tree::open(&workspace_root())
        .expect("Fix: run the tree contracts inside a git checkout of the repository.")
}

/// Every workspace member with its parsed manifest.
fn members(tree: &Tree) -> Vec<Member> {
    tree.member_manifests()
        .expect("Fix: every workspace member must declare a package name in a parsable manifest.")
}

/// The tree-relative root path each `[[test]]` row of one member names.
///
/// A row without `path` takes cargo's own default, `tests/<name>.rs`, so a row
/// spelled either way is judged against the file cargo would look for.
fn declared_roots(member: &Member) -> Vec<(String, String)> {
    member
        .manifest
        .get("test")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name").and_then(toml::Value::as_str)?;
            let relative = entry
                .get("path")
                .and_then(toml::Value::as_str)
                .map_or_else(|| format!("tests/{name}.rs"), str::to_string);
            Some((name.to_string(), format!("{}/{relative}", member.path)))
        })
        .collect()
}

/// Whether the member turned cargo's test autodiscovery off.
fn autodiscovery_is_off(member: &Member) -> bool {
    member
        .manifest
        .get("package")
        .and_then(|package| package.get("autotests"))
        .and_then(toml::Value::as_bool)
        == Some(false)
}

/// Every `.rs` file directly under one member's `tests/` directory.
fn test_files_directly_under(tree: &Tree, member: &Member) -> Vec<String> {
    let prefix = format!("{}/tests/", member.path);
    tree.paths()
        .iter()
        .filter_map(|path| path.to_str())
        .filter(|path| path.ends_with(".rs"))
        .filter(|path| {
            path.strip_prefix(&prefix)
                .is_some_and(|rest| !rest.contains('/'))
        })
        .map(str::to_string)
        .collect()
}

/// A row naming a file that is not there declares a target nothing can build.
#[test]
fn every_declared_test_target_names_a_file_that_exists() {
    let tree = tree();
    let mut dangling = Vec::new();
    for member in members(&tree) {
        for (name, root) in declared_roots(&member) {
            if !tree.absolute(&root).is_file() {
                dangling.push(format!("{}: [[test]] `{name}` names `{root}`", member.name));
            }
        }
    }

    assert!(
        dangling.is_empty(),
        "Fix: point each row at the file it should compile, or delete the row. \
         A [[test]] row naming no file builds nothing, and cargo reports it only \
         when the row's required-features are on:\n{}",
        dangling.join("\n")
    );
}

/// A file no target reaches holds assertions that never run.
///
/// Judged at the depth where no exclusion applies, through the same reader the
/// gate uses.
#[test]
fn every_test_file_of_a_member_without_autodiscovery_is_compiled_by_a_target() {
    let tree = tree();
    let mut unbuilt = Vec::new();
    for member in members(&tree) {
        if !autodiscovery_is_off(&member) {
            continue;
        }
        let compiled = ownership(&tree, &member).owners;
        for file in test_files_directly_under(&tree, &member) {
            if !compiled.contains_key(&file) {
                unbuilt.push(format!("{}: {file}", member.name));
            }
        }
    }

    assert!(
        unbuilt.is_empty(),
        "Fix: declare each file as a [[test]] row, or add it as a module of the \
         harness for its feature set. The package sets autotests = false, so \
         cargo compiles only what the manifest declares:\n{}",
        unbuilt.join("\n")
    );
}

/// The two contracts above judge a set the tree supplies, and it is not empty.
///
/// WHY: both pass when they judge nothing, and the way they come to judge
/// nothing is a member walk or a file walk that stopped resolving. Each
/// assertion here is a property of the grouped-harness layout rather than a
/// recorded count, so it holds as members and files come and go and fails when
/// a walk collapses.
#[test]
fn the_walk_judges_a_declared_set_the_tree_supplies() {
    let tree = tree();
    let members = members(&tree);
    let with_tests: Vec<&Member> = members
        .iter()
        .filter(|member| tree.absolute(&member.path).join("tests").is_dir())
        .collect();

    let discovering: Vec<&str> = with_tests
        .iter()
        .filter(|member| !autodiscovery_is_off(member))
        .map(|member| member.name.as_str())
        .collect();
    assert!(
        discovering.is_empty(),
        "Fix: set autotests = false. A member with a tests/ directory that leaves \
         autodiscovery on links each file as its own target as well as running it \
         inside a harness, so one test reports twice: {}",
        discovering.join(", ")
    );

    let silent: Vec<&str> = with_tests
        .iter()
        .filter(|member| declared_roots(member).is_empty())
        .map(|member| member.name.as_str())
        .collect();
    assert!(
        silent.is_empty(),
        "Fix: declare a [[test]] target. These members carry test files and \
         autotests = false, so they compile no test at all: {}",
        silent.join(", ")
    );

    let rows: usize = members
        .iter()
        .map(|member| declared_roots(member).len())
        .sum();
    let files: usize = with_tests
        .iter()
        .map(|member| test_files_directly_under(&tree, member).len())
        .sum();
    assert!(
        !with_tests.is_empty() && files > rows,
        "Fix: the walk reached {} member(s) with tests, {rows} [[test]] row(s) and \
         {files} test file(s). Grouped harnesses carry many files per row, so a \
         file count at or below the row count means the file walk stopped \
         resolving and the contracts above judged an empty set.",
        with_tests.len()
    );
}
