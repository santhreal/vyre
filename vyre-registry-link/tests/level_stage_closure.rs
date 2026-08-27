//! Every IR level has exactly one stage, and one owner per derived fact.
//!
//! WHY this suite exists: a level's pipeline says which passes act at that
//! level, and says nothing about what rejects a subject those passes must never
//! see, what form they converge to, or which facts are the level's to hold.
//! Each of those already exists somewhere in this workspace, spread across the
//! crate that owns the level's subject, so nothing could tell a level that owns
//! all three from a level that owns none.
//!
//! This crate is where the question can be asked. A stage is registered by the
//! crate that owns its subject, and only a binary linking every such crate
//! reads the whole registry; a suite living in one of them would read its own
//! rows and report a partial registry as the registry.
//!
//! The level set is parsed from `vyre-spec` source rather than read from
//! `IrLevel::all`, because a variant added to the enum and omitted from `all`
//! is exactly the silent hole this suite exists to close.
//!
//! What this does NOT catch: whether a stage's verifier is the right verifier
//! for its level. Each owning crate proves its own stage accepts a well-formed
//! subject and rejects a malformed one; that proof needs fixtures only the
//! owner has.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::optimizer::level_contract::{analysis_owners, stage_for_level, LevelVerdict};
use vyre_registry_link::level::live_level_stages;
use vyre_spec::IrLevel;

/// A type no stage owns, so every stage must refuse it.
#[derive(Debug)]
struct ForeignSubject;

/// The `IrLevel` variant names `vyre-spec` declares, read from source.
fn declared_level_variants() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join("ir_level.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} to derive the level set: {err}"));
    let body = vyre_test_support::braced_body(&source, "pub enum IrLevel {")
        .unwrap_or_else(|| panic!("Fix: {path:?} no longer declares `pub enum IrLevel`"));
    vyre_test_support::top_level_variant_names(body)
}

/// Adding an IR level turns this suite red until it has a stage.
#[test]
fn every_declared_ir_level_has_exactly_one_stage() {
    let declared = declared_level_variants();
    assert!(
        declared.len() >= 5,
        "Fix: the IrLevel source enumeration found only {} variants; the scan is broken, not the \
         enum",
        declared.len()
    );

    let by_level: BTreeMap<&str, Vec<&'static str>> =
        live_level_stages()
            .iter()
            .fold(BTreeMap::new(), |mut acc, stage| {
                acc.entry(stage.level().name())
                    .or_default()
                    .push(stage.subject());
                acc
            });

    for level in IrLevel::all() {
        let subjects = by_level.get(level.name()).map_or(&[][..], Vec::as_slice);
        assert_eq!(
            subjects.len(),
            1,
            "Fix: level `{}` has {} registered stages, not one: {subjects:?}. Register a stage in \
             the crate that owns the level's subject.",
            level.name(),
            subjects.len()
        );
        assert!(
            stage_for_level(*level).is_some(),
            "Fix: level `{}` resolves to no single stage",
            level.name()
        );
    }

    assert_eq!(
        live_level_stages().len(),
        declared.len(),
        "Fix: {} stages are registered for {} declared levels; a stage names a level the enum does \
         not declare, or a level has none",
        live_level_stages().len(),
        declared.len()
    );
}

/// The declared enum and the linked registry name the same levels.
#[test]
fn stage_levels_are_exactly_the_declared_variants() {
    let declared = declared_level_variants();
    let registered: BTreeSet<String> = live_level_stages()
        .iter()
        .map(|stage| format!("{:?}", stage.level()))
        .collect();
    assert_eq!(
        registered, declared,
        "Fix: the registered stage levels and the declared IrLevel variants differ"
    );
}

/// One level owns each derived fact set.
#[test]
fn every_analysis_has_one_owning_level() {
    let owners = analysis_owners();
    assert!(
        !owners.is_empty(),
        "Fix: no level declares an analysis; a level with no fact set has no analysis manager"
    );
    let mut seen: BTreeMap<&str, IrLevel> = BTreeMap::new();
    for (name, level) in owners {
        if let Some(previous) = seen.insert(name, level) {
            panic!(
                "Fix: analysis `{name}` is declared by both {previous:?} and {level:?}. One level \
                 owns a fact set; move the row or rename the fact."
            );
        }
    }
}

/// Every level declares at least one fact set of its own.
#[test]
fn every_stage_declares_an_analysis() {
    for stage in live_level_stages() {
        assert!(
            !stage.analyses().is_empty(),
            "Fix: level `{}` declares no analysis, so nothing states which facts it holds",
            stage.level().name()
        );
    }
}

/// A subject of the wrong type is refused, never verified.
#[test]
fn every_stage_refuses_a_foreign_subject() {
    let foreign = ForeignSubject;
    for stage in live_level_stages() {
        assert_eq!(
            stage.verify(&foreign),
            LevelVerdict::WrongSubject {
                expected: stage.subject()
            },
            "Fix: level `{}` verified a subject of another type",
            stage.level().name()
        );
        assert_eq!(
            stage.is_canonical(&foreign),
            LevelVerdict::WrongSubject {
                expected: stage.subject()
            },
            "Fix: level `{}` called a subject of another type canonical",
            stage.level().name()
        );
    }
}

/// Each level verifies its own subject type.
#[test]
fn stage_subjects_are_distinct() {
    let subjects: BTreeSet<&'static str> = live_level_stages()
        .iter()
        .map(|stage| stage.subject())
        .collect();
    assert_eq!(
        subjects.len(),
        live_level_stages().len(),
        "Fix: two levels verify the same subject type, so one of them is verifying another \
         level's IR"
    );
}
