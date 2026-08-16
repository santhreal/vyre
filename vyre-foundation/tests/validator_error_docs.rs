//! The class closed here: a published error catalog that describes a rule
//! which no longer runs, omits one that does, or carries a row that names no
//! invariant and offers no correction.
//!
//! The catalog used to be prose. A rule could be added, renamed or deleted in
//! source and the document stayed as it was until someone remembered it, and
//! four rows had decayed to "Program validation error 042" with the correction
//! "See diagnostic output", which is a row that exists and says nothing.
//!
//! `docs/generated/error-codes.toml` is now rendered from the rule table, so
//! the only way for the document to be wrong is for it to be stale, which is
//! what the first test measures. The other two hold the table itself to the
//! shape a registry has to have, because a generator faithfully reproduces a
//! defect in its input.

use std::collections::BTreeMap;
use std::fs;

use vyre_foundation::validate::{render_catalog_toml, rules};
use vyre_test_support::monorepo::vyre_workspace_root;

/// Path of the generated catalog, relative to the workspace root.
const CATALOG_PATH: &str = "docs/generated/error-codes.toml";

/// Set this to any value to rewrite the catalog instead of reporting drift.
const WRITE_ENV: &str = "VYRE_WRITE_ERROR_CATALOG";

#[test]
fn generated_catalog_matches_the_live_validation_rules() {
    let path = vyre_workspace_root().join(CATALOG_PATH);
    let rendered = render_catalog_toml();

    if std::env::var_os(WRITE_ENV).is_some() {
        fs::write(&path, &rendered).unwrap_or_else(|error| {
            panic!("cannot write {CATALOG_PATH}: {error}. Fix: check the path is writable.")
        });
    }

    let on_disk = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read {CATALOG_PATH}: {error}. \
             Fix: run the suite with {WRITE_ENV}=1 to generate it."
        )
    });
    if on_disk == rendered {
        return;
    }

    let findings = divergences(&on_disk, &rendered);
    assert!(
        findings.is_empty(),
        "{CATALOG_PATH} does not match the live validation rules:\n{}\n\
         Fix: run the suite with {WRITE_ENV}=1 to regenerate it.",
        findings.join("\n")
    );
    panic!(
        "{CATALOG_PATH} differs from the render outside any rule block. \
         Fix: run the suite with {WRITE_ENV}=1 to regenerate it."
    );
}

#[test]
fn every_rule_states_an_invariant_and_a_correction() {
    let mut findings = Vec::new();
    for rule in rules() {
        if rule.invariant.trim().is_empty() {
            findings.push(format!("{}: invariant is empty", rule.code));
        }
        if rule.corrective_action.trim().is_empty() {
            findings.push(format!("{}: corrective action is empty", rule.code));
        }
        if rule.invariant.contains("Program validation error") {
            findings.push(format!(
                "{}: invariant restates the code instead of naming the invariant",
                rule.code
            ));
        }
        if rule.corrective_action.contains("See diagnostic output")
            || rule.corrective_action.contains("See error message")
        {
            findings.push(format!(
                "{}: corrective action defers to the diagnostic instead of naming a correction",
                rule.code
            ));
        }
    }
    assert!(
        findings.is_empty(),
        "the rule table carries rows that say nothing:\n{}\n\
         Fix: state the invariant and the correction in \
         vyre-foundation/src/validate/catalog.rs, taking the wording from the \
         emission site.",
        findings.join("\n")
    );
}

#[test]
fn rule_codes_are_unique_and_ordered() {
    let codes: Vec<&str> = rules().iter().map(|rule| rule.code).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        codes, sorted,
        "the rule table must list each code once, in ascending order. \
         Fix: insert the new rule at its sorted position in \
         vyre-foundation/src/validate/catalog.rs."
    );
}

/// Path of the hand-written reference page that counts the catalog.
const REFERENCE_PATH: &str = "docs/reference/diagnostics.md";

#[test]
fn the_diagnostics_reference_counts_the_rules_it_describes() {
    let path = vyre_workspace_root().join(REFERENCE_PATH);
    let page = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("cannot read {REFERENCE_PATH}: {error}. Fix: check the path exists.")
    });

    let mut per_phase: BTreeMap<&str, usize> = BTreeMap::new();
    for rule in rules() {
        *per_phase.entry(rule.phase.as_str()).or_default() += 1;
    }

    let total = rules().len();
    assert!(
        page.contains(&format!("rule set: {total}\n")),
        "{REFERENCE_PATH} does not state the live rule count of {total}. \
         Fix: correct the sentence that counts the catalog."
    );

    let documented: BTreeMap<&str, usize> = validation_section(&page)
        .lines()
        .filter_map(phase_row)
        .collect();
    assert_eq!(
        documented, per_phase,
        "the phase table in {REFERENCE_PATH} does not match the live catalog. \
         Fix: correct one row per divergent phase; the catalog is the authority."
    );
}

/// The `## Validation codes` section, which is the only place the page counts
/// validation rules. The backend error table lives in another section and its
/// rows are not phase rows.
fn validation_section(page: &str) -> &str {
    let start = page
        .find("\n## Validation codes\n")
        .unwrap_or_else(|| panic!("{REFERENCE_PATH} has no validation codes section"));
    let rest = &page[start + 1..];
    match rest[1..].find("\n## ") {
        Some(end) => &rest[..end + 1],
        None => rest,
    }
}

/// Read a `| phase | count |` row, skipping the header and the rule that draws
/// it.
fn phase_row(line: &str) -> Option<(&str, usize)> {
    let mut cells = line.trim().strip_prefix('|')?.split('|');
    let phase = cells.next()?.trim();
    let count = cells.next()?.trim().parse().ok()?;
    cells.next()?;
    if cells.next().is_some() {
        return None;
    }
    Some((phase, count))
}

/// One finding per rule code whose block differs, plus one per code present on
/// only one side.
fn divergences(on_disk: &str, rendered: &str) -> Vec<String> {
    let disk_blocks = rule_blocks(on_disk);
    let live_blocks = rule_blocks(rendered);
    let mut findings = Vec::new();
    for (code, live) in &live_blocks {
        match disk_blocks.get(code) {
            None => findings.push(format!("{code}: rule is live but absent from the catalog")),
            Some(disk) if disk != live => findings.push(format!(
                "{code}: catalog row differs from the live rule\n  catalog: {disk}\n  live:    {live}"
            )),
            Some(_) => {}
        }
    }
    for code in disk_blocks.keys() {
        if !live_blocks.contains_key(code) {
            findings.push(format!("{code}: catalog row names a rule that is not live"));
        }
    }
    findings.sort();
    findings
}

/// Split a rendered catalog into one single-line body per rule code.
fn rule_blocks(document: &str) -> BTreeMap<String, String> {
    let mut blocks = BTreeMap::new();
    for block in document.split("[[rule]]").skip(1) {
        let body = block.trim();
        let Some(code) = field(body, "code") else {
            continue;
        };
        blocks.insert(code, body.replace('\n', " | "));
    }
    blocks
}

/// Read one `name = "value"` field out of a rule block.
fn field(block: &str, name: &str) -> Option<String> {
    let prefix = format!("{name} = \"");
    block.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(&prefix)?;
        rest.strip_suffix('"').map(ToOwned::to_owned)
    })
}
