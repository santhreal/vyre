use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::records::{
    HygieneFindingClass, PanicBudgetArtifact, PanicBudgetDocument, PanicBudgetRow,
    PANIC_BUDGET_SCHEMA_VERSION, PANIC_BUDGET_SOURCE,
};
use super::threshold_policy::relative_to_vyre;

/// Whether a classified finding is a panic that nothing else answers for.
///
/// Read off the classification rather than re-scanning, so the population is the
/// one the artifact records. A documented contract is a different pattern by the
/// time it reaches here, and a hot-path panic is already a release blocker, so
/// neither is counted twice.
pub(crate) fn is_unbounded_panic(class: &HygieneFindingClass) -> bool {
    matches!(class.pattern, "panic_macro" | "unwrap_call" | "expect_call")
        && !class.release_blocker
        && matches!(class.surface, "production" | "release_tooling")
}

/// The crate a scanned path belongs to.
///
/// The first path component, which is the crate directory for every workspace
/// member and the containing directory for the two nested ones. Derived from the
/// path so a new crate needs no edit here to be counted.
pub(crate) fn crate_of_path(path: &str) -> String {
    path.replace('\\', "/")
        .split('/')
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Hold the undocumented panic population to the recorded per-crate ceiling.
///
/// Every failure path returns a blocker rather than an empty budget, because a
/// budget that could not be read must not read as a tree that owes nothing.
pub(crate) fn collect_panic_budget(
    vyre_root: &Path,
    classes: &[HygieneFindingClass],
) -> PanicBudgetArtifact {
    let mut measured = BTreeMap::<String, usize>::new();
    for class in classes.iter().filter(|class| is_unbounded_panic(class)) {
        let relative = relative_to_vyre(vyre_root, Path::new(&class.path));
        *measured.entry(crate_of_path(&relative)).or_insert(0) += 1;
    }

    let mut artifact = PanicBudgetArtifact {
        schema_version: PANIC_BUDGET_SCHEMA_VERSION,
        source: PANIC_BUDGET_SOURCE,
        rows: Vec::new(),
        unrecorded: Vec::new(),
        notes: Vec::new(),
        blockers: Vec::new(),
    };

    let path = vyre_root.join(PANIC_BUDGET_SOURCE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} could not be read ({error}), so {} undocumented panic(s) outside the release surface are bounded by nothing",
                measured.values().sum::<usize>()
            ));
            return artifact;
        }
    };
    let document = match toml::from_str::<PanicBudgetDocument>(&text) {
        Ok(document) => document,
        Err(error) => {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} is not readable as a panic budget: {error}"
            ));
            return artifact;
        }
    };
    if document.schema != PANIC_BUDGET_SCHEMA_VERSION {
        artifact.blockers.push(format!(
            "{PANIC_BUDGET_SOURCE} declares schema {} against {PANIC_BUDGET_SCHEMA_VERSION}",
            document.schema
        ));
        return artifact;
    }

    let mut ceilings = BTreeMap::<String, usize>::new();
    for row in document.crate_budget {
        if let Some(previous) = ceilings.insert(row.name.clone(), row.ceiling) {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} records {} twice, at {previous} and {}, so one ceiling is unread",
                row.name, row.ceiling
            ));
        }
    }

    for (crate_name, count) in &measured {
        match ceilings.get(crate_name) {
            Some(ceiling) if count > ceiling => artifact.blockers.push(format!(
                "{crate_name} carries {count} undocumented panic(s) outside the release surface against a ceiling of {ceiling}: document the contract in a `# Panics` section, return an error instead, or delete the panic"
            )),
            Some(ceiling) if count < ceiling => artifact.notes.push(format!(
                "{crate_name} carries {count} undocumented panic(s) against a ceiling of {ceiling}: lower the ceiling in {PANIC_BUDGET_SOURCE} to {count}, because a ceiling above the tree covers the next panic added to it"
            )),
            Some(_) => {}
            None => {
                artifact.unrecorded.push(crate_name.clone());
                artifact.blockers.push(format!(
                    "{crate_name} carries {count} undocumented panic(s) outside the release surface and {PANIC_BUDGET_SOURCE} records no ceiling for it"
                ));
            }
        }
    }
    for (crate_name, ceiling) in &ceilings {
        if !measured.contains_key(crate_name) && *ceiling > 0 {
            artifact.blockers.push(format!(
                "{PANIC_BUDGET_SOURCE} records a ceiling of {ceiling} for {crate_name}, which now carries none: lower the row to 0, because the ceiling is what stands between the crate and the next panic added to it"
            ));
        }
    }

    artifact.rows = ceilings
        .into_iter()
        .map(|(crate_name, ceiling)| PanicBudgetRow {
            measured: measured.get(&crate_name).copied().unwrap_or_default(),
            crate_name,
            ceiling,
        })
        .collect();
    artifact
}
