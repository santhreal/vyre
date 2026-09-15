//! The generated code catalog agrees with the code that emits it.
//!
//! WHY: the previous version of this file parsed a markdown table for the
//! variant-to-id mapping and walked a hand-written list of seven variants. The
//! enum had nine. `CooperativeResidencyExceeded` and `DeviceLost` were absent
//! from the list, so no test covered them, and the id-comparison test skipped
//! any variant the catalog did not mention (`else { continue }`) — the one case
//! it existed to catch. The markdown was then deleted outright and the test
//! failed on a missing file rather than on a wrong id.
//!
//! The list is now `ErrorCode::ALL`, a const assertion in `backend/error.rs`
//! holds its order to `stable_id`, and
//! `every_declared_variant_is_catalogued` below reads the `pub enum ErrorCode`
//! declaration at run time and holds the membership to it. A variant added to
//! the enum and not to `ALL` fails that test; a variant added to `ALL` and not
//! to `summary` does not compile. What remains is that the committed catalog
//! matches the rendered one, which the first test does.
//!
//! Not covered here: whether an id was renumbered between releases. The
//! committed file makes such a change visible in review; nothing at test time
//! remembers a previous release's numbering.

use std::fs;
use std::path::PathBuf;

use vyre_driver::error_catalog::render_catalog_toml;
use vyre_driver::ErrorCode;
use vyre_driver::DEPRECATED_OP_CODE;

fn catalog_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("docs/generated/driver-error-codes.toml")
}

#[test]
fn committed_catalog_matches_the_rendered_one() {
    let rendered = render_catalog_toml();
    let path = catalog_path();

    let committed = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "code catalog: cannot read {}: {err}. Regenerate with \
             `./cargo_full run -p xtask --bin xtask -- error-codes --write`",
            path.display()
        )
    });

    // Compared line by line, not byte by byte. A checkout that materializes the
    // committed file with CRLF endings is not a divergent catalog, and a byte
    // comparison reported one while the diff below found nothing to name: the
    // failure read "0 divergent rows" on every Windows run.
    let committed_lines: Vec<&str> = committed.lines().collect();
    let rendered_lines: Vec<&str> = rendered.lines().collect();
    if committed_lines == rendered_lines {
        return;
    }

    let mut findings = Vec::new();
    for (index, (have, want)) in committed_lines.iter().zip(&rendered_lines).enumerate() {
        if have != want {
            findings.push(format!(
                "line {}: committed {have:?}, source {want:?}",
                index + 1
            ));
        }
    }
    match committed_lines.len().cmp(&rendered_lines.len()) {
        std::cmp::Ordering::Less => findings.push(format!(
            "committed catalog is {} lines short",
            rendered_lines.len() - committed_lines.len()
        )),
        std::cmp::Ordering::Greater => findings.push(format!(
            "committed catalog has {} lines the source does not produce",
            committed_lines.len() - rendered_lines.len()
        )),
        std::cmp::Ordering::Equal => {}
    }

    panic!(
        "code catalog: {} divergent rows.\n{}\nRegenerate with \
         `./cargo_full run -p xtask --bin xtask -- error-codes --write`",
        findings.len(),
        findings.join("\n")
    );
}

#[test]
fn every_variant_is_catalogued_with_its_stable_id_and_a_description() {
    let rendered = render_catalog_toml();
    let table: toml::Table = toml::from_str(&rendered).expect("rendered catalog parses as TOML");

    let rows = table["backend_error"]
        .as_array()
        .expect("backend_error is an array of tables");
    assert_eq!(
        rows.len(),
        ErrorCode::ALL.len(),
        "catalog holds {} backend rows for {} variants",
        rows.len(),
        ErrorCode::ALL.len()
    );

    let mut seen_ids = Vec::new();
    for (row, code) in rows.iter().zip(ErrorCode::ALL) {
        let variant = row["variant"].as_str().expect("variant is a string");
        let id = row["id"].as_integer().expect("id is an integer");
        let summary = row["summary"].as_str().expect("summary is a string");

        assert_eq!(
            variant,
            format!("{code:?}"),
            "row order follows ErrorCode::ALL"
        );
        assert_eq!(
            u32::try_from(id).expect("id fits u32"),
            code.stable_id(),
            "catalog id for {variant} disagrees with the binary"
        );
        assert!(
            summary.len() > 20 && summary.ends_with('.'),
            "{variant} carries a placeholder description: {summary:?}"
        );
        assert!(
            !seen_ids.contains(&id),
            "id {id} is assigned to more than one variant"
        );
        seen_ids.push(id);
    }
}

#[test]
fn the_deprecation_warning_is_catalogued() {
    let rendered = render_catalog_toml();
    let table: toml::Table = toml::from_str(&rendered).expect("rendered catalog parses as TOML");

    let codes: Vec<&str> = table["diagnostic"]
        .as_array()
        .expect("diagnostic is an array of tables")
        .iter()
        .map(|row| row["code"].as_str().expect("code is a string"))
        .collect();

    assert!(
        codes.contains(&DEPRECATED_OP_CODE),
        "the only non-validation diagnostic code this crate emits is uncatalogued: \
         have {codes:?}"
    );
}

/// Every variant the enum declares reaches the catalog.
///
/// WHY: the member set is read from source rather than written here. A
/// hardwritten roster is what let `CooperativeResidencyExceeded` and
/// `DeviceLost` go uncovered for as long as they did, and no const assertion
/// can prove completeness for an enum: a new variant absent from `ALL` is
/// never evaluated by the const block that walks `ALL`.
///
/// Not covered: a variant declared behind `cfg`. The scan reads text, so it
/// reports every variant in the declaration whichever features the runner
/// selects, which is the stronger direction for this table.
#[test]
fn every_declared_variant_is_catalogued() {
    let path =
        vyre_test_support::monorepo::vyre_workspace_root().join("vyre-driver/src/backend/error.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cannot read the ErrorCode declaration at {path:?}: {err}"));
    let body =
        vyre_test_support::braced_body(&source, "pub enum ErrorCode {").unwrap_or_else(|| {
            panic!("no `pub enum ErrorCode` declaration in {path:?}; update this enumeration")
        });
    let declared = vyre_test_support::top_level_variant_names(body);

    assert!(
        declared.len() >= 8,
        "the variant scan found {} names, which is a broken scan rather than a small enum",
        declared.len()
    );

    let catalogued: std::collections::BTreeSet<String> = ErrorCode::ALL
        .iter()
        .map(|code| format!("{code:?}"))
        .collect();
    assert_eq!(
        declared, catalogued,
        "ErrorCode::ALL and the ErrorCode declaration disagree; every variant must be catalogued"
    );
}
