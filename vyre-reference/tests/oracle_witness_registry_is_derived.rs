//! WHY: closes the class where a composition family is added to the operation
//! registry and ships with no independent witness, because nothing asked the
//! question. `composition_witness` published a few hundred free functions and
//! no structure said which family any of them served, so "does every family
//! have a witness" had no answer to be wrong.
//!
//! The family vocabulary is derived here from
//! `docs/generated/op-inventory.toml`, which `cargo xtask list-ops --write`
//! generates from the live operation registry. Every category it names must
//! have a row in [`COMPOSITION_WITNESS_FAMILIES`], and every row must name a
//! category that file still declares. A new category therefore turns this
//! suite RED until someone records either the module that witnesses it or the
//! reason it carries no witness.
//!
//! The reverse direction is held too: a `composition_witness` module that no
//! family names is dead code unless it is recorded in
//! [`MODULES_WITHOUT_A_CATALOG_FAMILY`] with the reason.
//!
//! What this does NOT catch: whether a family's witness is mathematically
//! right, or whether it covers every operation in the family. A row proves a
//! decision was recorded and the module is reachable; the per-family
//! known-answer contracts in `composition_witness_*_contracts.rs` prove the
//! arithmetic.

use std::collections::{BTreeMap, BTreeSet};

use vyre_reference::composition_witness::{
    witness_families, witness_family, FamilyWitness, COMPOSITION_WITNESS_FAMILIES,
    MODULES_WITHOUT_A_CATALOG_FAMILY,
};
use vyre_test_support::monorepo::vyre_workspace_root;
use vyre_test_support::read_source_file_bounded;

/// Generated inventory of every registered operation.
const OP_INVENTORY: &str = "docs/generated/op-inventory.toml";

/// Fewest categories a working inventory scan finds.
///
/// A scan that read the wrong file, or a parse that matched nothing, reports
/// an empty family set, and an empty set is covered by an empty registry. The
/// floor sits below the current count so it catches a broken derivation
/// without an edit whenever the vocabulary grows.
const FAMILY_FLOOR: usize = 20;

/// Fewest operations a working inventory scan finds.
const OPERATION_FLOOR: usize = 300;

/// Every operation category the generated inventory declares.
fn declared_families() -> BTreeSet<String> {
    let path = vyre_workspace_root().join(OP_INVENTORY);
    let source = read_source_file_bounded(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {OP_INVENTORY}: {err}"));
    let operations = source.matches("[[operation]]").count();
    assert!(
        operations >= OPERATION_FLOOR,
        "Fix: {OP_INVENTORY} declares {operations} operations, below the floor of \
         {OPERATION_FLOOR}. A short inventory covers a short family set vacuously."
    );
    source
        .lines()
        .filter_map(|line| line.strip_prefix("category = \""))
        .filter_map(|rest| rest.strip_suffix('"'))
        .map(str::to_string)
        .collect()
}

/// Every module `composition_witness` declares.
fn declared_witness_modules() -> BTreeSet<String> {
    let directory = vyre_workspace_root().join("vyre-reference/src/composition_witness");
    let entries = std::fs::read_dir(&directory)
        .unwrap_or_else(|err| panic!("Fix: cannot read {directory:?}: {err}"));
    let mut modules = BTreeSet::new();
    for entry in entries {
        let path = entry.expect("Fix: unreadable directory entry").path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let stem = path
                .file_stem()
                .expect("Fix: a .rs path with no file stem")
                .to_string_lossy()
                .to_string();
            if stem != "mod" && stem != "registry" {
                modules.insert(stem);
            }
        }
    }
    modules
}

/// Modules `composition_witness` re-exports witnesses from.
///
/// A submodule such as `graph_dominator` reaches consumers through its parent
/// `graph`, so the parent is what a family row names. The set is read from the
/// `pub use <module>::{...}` declarations rather than listed here.
fn re_exporting_modules() -> BTreeSet<String> {
    let path = vyre_workspace_root().join("vyre-reference/src/composition_witness/mod.rs");
    let source = read_source_file_bounded(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read the witness module root: {err}"));
    source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub use "))
        .filter_map(|rest| rest.split("::").next())
        .filter(|name| {
            !name.is_empty()
                && name != &"registry"
                && name
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        })
        .map(str::to_string)
        .collect()
}

/// Every declared composition family has a registry row, and every row names a
/// declared family.
#[test]
fn every_declared_family_has_a_registry_row() {
    let families = declared_families();
    assert!(
        families.len() >= FAMILY_FLOOR,
        "Fix: the family derivation found {} categories in {OP_INVENTORY}, below the floor of \
         {FAMILY_FLOOR}. A derivation that finds nothing certifies nothing.",
        families.len()
    );

    let rows: BTreeMap<&str, &FamilyWitness> = COMPOSITION_WITNESS_FAMILIES
        .iter()
        .map(|row| (row.family, &row.witness))
        .collect();
    assert_eq!(
        rows.len(),
        COMPOSITION_WITNESS_FAMILIES.len(),
        "Fix: COMPOSITION_WITNESS_FAMILIES names the same family twice; one row per family."
    );

    let missing: Vec<&String> = families
        .iter()
        .filter(|family| !rows.contains_key(family.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: the operation registry declares the composition families {missing:?} and the \
         witness registry has no row for them. Add a COMPOSITION_WITNESS_FAMILIES row naming the \
         witness module, or record why the family carries no independent witness."
    );

    let stale: Vec<&str> = rows
        .keys()
        .copied()
        .filter(|family| !families.contains(*family))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: COMPOSITION_WITNESS_FAMILIES has rows for {stale:?}, which {OP_INVENTORY} no longer \
         declares. Drop the rows so the registry stays a statement about the current vocabulary."
    );
}

/// Every witnessed row names a real re-exporting module, and every absent row
/// carries a reason.
#[test]
fn every_registry_row_names_a_real_witness_owner() {
    let modules = re_exporting_modules();
    assert!(
        modules.len() >= 10,
        "Fix: the witness module derivation found only {modules:?}; a short module set makes \
         every ownership assertion vacuous."
    );

    let mut failures = Vec::new();
    for row in witness_families() {
        match row.witness {
            FamilyWitness::Owned { module, .. } => {
                if !modules.contains(module) {
                    failures.push(format!(
                        "family `{}` names witness module `{module}`, which composition_witness \
                         does not re-export from",
                        row.family
                    ));
                }
            }
            FamilyWitness::Absent { reason } => {
                if reason.trim().is_empty() {
                    failures.push(format!(
                        "family `{}` is recorded as unwitnessed with no reason",
                        row.family
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: every witness registry row must resolve to a real owner.\n{}",
        failures.join("\n")
    );
}

/// Every witness module is reached by a family, or recorded as reached by none.
///
/// Without this the registry closes one direction only: a module could be
/// added, wired to nothing, and stay green because no family happened to name
/// it.
#[test]
fn every_witness_module_is_named_or_recorded() {
    let declared = declared_witness_modules();
    let re_exported = re_exporting_modules();
    let unknown: Vec<&String> = re_exported
        .iter()
        .filter(|module| !declared.contains(*module))
        .collect();
    assert!(
        unknown.is_empty(),
        "Fix: composition_witness re-exports from {unknown:?}, which is not a module file in that \
         directory."
    );

    let named: BTreeSet<&str> = COMPOSITION_WITNESS_FAMILIES
        .iter()
        .filter_map(|row| row.module())
        .collect();
    let recorded: BTreeMap<&str, &str> = MODULES_WITHOUT_A_CATALOG_FAMILY.iter().copied().collect();

    let orphans: Vec<&String> = re_exported
        .iter()
        .filter(|module| {
            !named.contains(module.as_str()) && !recorded.contains_key(module.as_str())
        })
        .collect();
    assert!(
        orphans.is_empty(),
        "Fix: the witness modules {orphans:?} are re-exported and no composition family names \
         them. Add the family row that uses them, or record the module in \
         MODULES_WITHOUT_A_CATALOG_FAMILY with the reason no category does."
    );

    let stale: Vec<&str> = recorded
        .keys()
        .copied()
        .filter(|module| !re_exported.contains(*module) || named.contains(*module))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: MODULES_WITHOUT_A_CATALOG_FAMILY records {stale:?}, which is either gone or now \
         named by a family row. Drop the row."
    );

    let empty_reason: Vec<&str> = MODULES_WITHOUT_A_CATALOG_FAMILY
        .iter()
        .filter(|(_, reason)| reason.trim().is_empty())
        .map(|&(module, _)| module)
        .collect();
    assert!(
        empty_reason.is_empty(),
        "Fix: {empty_reason:?} is recorded as reaching no family with no reason."
    );
}

/// Every witnessed family's probe reaches real witness code and is
/// deterministic.
///
/// A registry row whose probe cannot run is a name, not a witness owner. Two
/// calls agreeing also rejects a probe that reads ambient state, which would
/// make a family's witness output depend on run order.
#[test]
fn every_family_probe_is_reachable_and_deterministic() {
    let mut digests: BTreeMap<&str, u64> = BTreeMap::new();
    for row in witness_families() {
        let Some(probe) = row.probe() else { continue };
        let first = probe();
        let second = probe();
        assert_eq!(
            first, second,
            "Fix: the witness probe for family `{}` returned {first} then {second}. A witness \
             whose output depends on ambient state cannot judge an implementation.",
            row.family
        );
        digests.insert(row.family, first);
    }
    assert!(
        digests.len() >= FAMILY_FLOOR - MODULES_WITHOUT_A_CATALOG_FAMILY.len() - 4,
        "Fix: only {} families ran a probe; the registry has lost its witnessed rows.",
        digests.len()
    );

    let owners: BTreeMap<&str, BTreeSet<u64>> =
        COMPOSITION_WITNESS_FAMILIES
            .iter()
            .fold(BTreeMap::new(), |mut acc, row| {
                if let (Some(module), Some(digest)) =
                    (row.module(), digests.get(row.family).copied())
                {
                    acc.entry(module).or_default().insert(digest);
                }
                acc
            });
    for (module, module_digests) in &owners {
        assert_eq!(
            module_digests.len(),
            1,
            "Fix: two families naming witness module `{module}` produced different probe \
             digests {module_digests:?}, so one of them is not calling that module."
        );
    }
    assert!(
        owners
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            > 1,
        "Fix: every family probe produced the same digest, so the probes are not reaching \
         distinct witnesses."
    );
}

/// The lookup answers for a declared family and refuses an undeclared one.
#[test]
fn family_lookup_resolves_only_declared_families() {
    let bitset = witness_family("bitset").expect("Fix: the registry lost the bitset family row");
    assert!(
        matches!(bitset.witness, FamilyWitness::Owned { module, .. } if module == "bitset"),
        "Fix: the bitset family no longer names the bitset witness module."
    );
    assert!(
        witness_family("no-such-family").is_none(),
        "Fix: the registry resolved a family the operation registry never declared."
    );
}
