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
//! The list is now `ErrorCode::ALL`, and a const assertion in
//! `backend/error.rs` makes a variant missing from it a compile error. What
//! remains for a test to prove is that the committed catalog matches the
//! rendered one, which is what this file does.
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

    if committed == rendered {
        return;
    }

    let mut findings = Vec::new();
    let committed_lines: Vec<&str> = committed.lines().collect();
    let rendered_lines: Vec<&str> = rendered.lines().collect();
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
