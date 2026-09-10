//! What the verification budget must catch, and what it must not report.

use super::*;

/// A copied contract body compiled by two packages is duplication; two copies
/// inside one package are not.
///
/// The measure exists to catch a shared body textually compiled by several
/// packages, which is the shape that drifts and is absent from each consuming
/// archive. A package that repeats a fixture inside itself compiles it once and
/// owns both copies, so reporting it here would price a different defect under
/// this name.
#[test]
fn duplication_is_counted_across_packages_and_not_within_one() {
    let body = "fn contract() { assert!(true); }";
    let sources = vec![
        source("a/tests/one.rs", "package-a", body),
        source("a/tests/two.rs", "package-a", body),
    ];
    assert_eq!(
        duplicate_bytes_by_package(&sources),
        BTreeMap::new(),
        "Fix: two copies inside one package are compiled once and must not count as cross-package duplication."
    );

    let sources = vec![
        source("a/tests/one.rs", "package-a", body),
        source("b/tests/one.rs", "package-b", body),
    ];
    let counted = duplicate_bytes_by_package(&sources);
    assert_eq!(
        counted.get("package-b").copied(),
        Some(body.len() as u64),
        "Fix: a body compiled by a second package is that package's duplication; counted={counted:?}"
    );
    assert!(
        !counted.contains_key("package-a"),
        "Fix: the first occurrence in path order is the original and must not be counted."
    );
}

/// Normalization must see through formatting, not through an edit.
#[test]
fn duplication_survives_reformatting_and_not_a_changed_assertion() {
    let sources = vec![
        source("a/tests/one.rs", "package-a", "fn contract() {\n\n    // note\n    assert!(true);\n}"),
        source("b/tests/one.rs", "package-b", "fn contract() {\nassert!(true);\n}"),
    ];
    assert!(
        duplicate_bytes_by_package(&sources).contains_key("package-b"),
        "Fix: blank lines and line comments must not hide a copy."
    );

    let sources = vec![
        source("a/tests/one.rs", "package-a", "fn contract() { assert!(true); }"),
        source("b/tests/one.rs", "package-b", "fn contract() { assert!(false); }"),
    ];
    assert!(
        duplicate_bytes_by_package(&sources).is_empty(),
        "Fix: an edited copy is a different body here; `dup-scan` owns near-duplicate measurement."
    );
}

/// A `#[path]` include is judged by where it resolves, not by whether it climbs.
///
/// `../support/x.rs` from `pkg/tests/deep/case.rs` stays inside `pkg`, and
/// `../../shared/x.rs` from `pkg/tests/case.rs` does not. A rule that reported
/// every `..` would convict the first and a rule that looked for a leading
/// `../..` would miss a deeper escape.
#[test]
fn a_path_include_is_judged_by_where_it_lands() {
    assert!(
        !resolves_outside("pkg/tests/deep/case.rs", "../support/x.rs", "pkg"),
        "Fix: an include that climbs and stays inside the package is not an escape."
    );
    assert!(
        resolves_outside("pkg/tests/case.rs", "../../shared/x.rs", "pkg"),
        "Fix: an include resolving above the package directory is an escape."
    );
    assert!(
        resolves_outside("pkg/tests/deep/case.rs", "../../../other/x.rs", "pkg"),
        "Fix: a deeper escape must be caught wherever the file sits."
    );
    assert!(
        !resolves_outside("pkg/tests/case.rs", "./support/x.rs", "pkg"),
        "Fix: a sibling include is not an escape."
    );
    assert!(
        !resolves_outside("pkg/tests/case.rs", "support/x.rs", "pkg"),
        "Fix: a bare relative include is not an escape."
    );
}

/// A package directory that prefixes another must not claim its files.
///
/// `conform/vyre-conform` and `conform/vyre-conform-spec` share a prefix, and
/// matching on the shorter one first would attribute the longer package's test
/// bytes to its neighbour, so both tiers would be wrong and neither would say
/// so.
#[test]
fn the_longest_member_directory_owns_the_file() {
    let members = BTreeMap::from([
        ("conform/vyre-conform".to_string(), "vyre-conform".to_string()),
        (
            "conform/vyre-conform-spec".to_string(),
            "vyre-conform-spec".to_string(),
        ),
    ]);
    assert_eq!(
        owning_member("conform/vyre-conform-spec/tests/case.rs", &members)
            .map(|(package, _)| package),
        Some("vyre-conform-spec".to_string()),
        "Fix: the longest matching member directory owns the file."
    );
    assert_eq!(
        owning_member("conform/vyre-conform/tests/case.rs", &members).map(|(package, _)| package),
        Some("vyre-conform".to_string()),
        "Fix: the shorter package must still own its own files."
    );
    assert_eq!(
        owning_member("elsewhere/tests/case.rs", &members),
        None,
        "Fix: a file under no member belongs to no package."
    );
}

/// A count that disagrees with its row fails in both directions.
///
/// Under the row is the failure that looks harmless and is not: a row above the
/// tree covers the next target added to the tier instead of reporting it, which
/// is the same as having no row for as many targets as the gap is wide.
#[test]
fn a_count_below_its_row_fails_as_loudly_as_one_above() {
    assert!(
        exact("libraries", "compile_units", 40, 40).is_none(),
        "Fix: a count equal to its row is clean."
    );

    let above = exact("libraries", "compile_units", 41, 40)
        .expect("Fix: a count above its row must report.");
    let below = exact("libraries", "compile_units", 39, 40)
        .expect("Fix: a count below its row must report.");
    assert!(
        above.fix.contains("or remove the targets"),
        "Fix: a count above its row must name recording the decision or removing the targets; fix={}",
        above.fix
    );
    assert!(
        below.fix.contains("--lower-budget libraries"),
        "Fix: a count below its row must name the command that lowers it; fix={}",
        below.fix
    );
}

/// `--lower-budget` refuses to raise any column.
///
/// A lowering command that silently raises a neighbouring column is how a
/// ratchet stops ratcheting, and nothing downstream would report it because the
/// file it writes is the authority everything else reads.
#[test]
fn lowering_refuses_to_raise_a_neighbouring_column() {
    let pin = TierCost {
        compile_units: 40,
        link_units: 60,
        duplicate_source_bytes: 0,
        disk_use_bytes: 1_000,
    };
    let raises_one_column = TierCost {
        compile_units: 39,
        link_units: 61,
        duplicate_source_bytes: 0,
        disk_use_bytes: 900,
    };
    assert!(
        raises_any_column(&raises_one_column, &pin),
        "Fix: a measurement above its row in any column must refuse the lowering."
    );
    let lowers_every_column = TierCost {
        compile_units: 39,
        link_units: 59,
        duplicate_source_bytes: 0,
        disk_use_bytes: 900,
    };
    assert!(
        !raises_any_column(&lowers_every_column, &pin),
        "Fix: a measurement at or below its row in every column is lowerable."
    );
}

/// A tier the ownership record declares and the budget does not is a finding,
/// and so is the reverse.
///
/// A tier added to the ownership record must turn this red until somebody
/// records what it costs. Reading only the declared rows would let a whole
/// layer of crates arrive unmeasured and the sweep stay green.
#[test]
fn a_tier_is_reported_when_either_side_names_it_alone() {
    let measurement = Measurement {
        schema_version: SCHEMA_VERSION,
        tiers: vec![TierRow {
            tier: "libraries".to_string(),
            packages: vec!["vyre-libs-math".to_string()],
            cost: TierCost::default(),
        }],
        shared_contracts: Vec::new(),
        duplicated_shared_contracts: Vec::new(),
        escaping_path_includes: Vec::new(),
    };
    let declared = BTreeMap::from([("emitter".to_string(), TierCost::default())]);
    let findings = judge(&measurement, &declared);
    let messages = findings
        .iter()
        .map(|finding| finding.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        messages.contains("`emitter` has a budget row and the ownership record declares no such tier"),
        "Fix: a stale row must be reported; messages={messages}"
    );
    assert!(
        messages.contains("tier `libraries`") && messages.contains("no budget row"),
        "Fix: a tier with no row must be reported; messages={messages}"
    );
}

/// Disk use is a ceiling, so under it is clean and over it is not.
#[test]
fn disk_use_is_a_ceiling_and_the_counts_are_not() {
    let measurement = |disk: u64| Measurement {
        schema_version: SCHEMA_VERSION,
        tiers: vec![TierRow {
            tier: "libraries".to_string(),
            packages: Vec::new(),
            cost: TierCost {
                compile_units: 1,
                link_units: 1,
                duplicate_source_bytes: 0,
                disk_use_bytes: disk,
            },
        }],
        shared_contracts: Vec::new(),
        duplicated_shared_contracts: Vec::new(),
        escaping_path_includes: Vec::new(),
    };
    let declared = BTreeMap::from([(
        "libraries".to_string(),
        TierCost {
            compile_units: 1,
            link_units: 1,
            duplicate_source_bytes: 0,
            disk_use_bytes: 1_000,
        },
    )]);
    assert!(
        judge(&measurement(900), &declared).is_empty(),
        "Fix: disk use below its ceiling is clean."
    );
    let over = judge(&measurement(1_001), &declared);
    assert_eq!(over.len(), 1, "Fix: disk use above its ceiling must report; findings={over:?}");
    assert!(
        over[0].message.contains("against a ceiling of 1000"),
        "Fix: the finding must state the ceiling; message={}",
        over[0].message
    );
}

/// Build one test source for the duplication measure.
fn source(path: &str, package: &str, text: &str) -> TestSource {
    TestSource {
        path: path.to_string(),
        package: package.to_string(),
        bytes: text.len() as u64,
        normalized: normalize(text),
    }
}
