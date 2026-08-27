//! The level-stage registry, with every crate that submits into it linked.
//!
//! A level stage lives in the crate that owns the level's subject: the logical
//! levels here in the foundation, the physical kernel in the lowering, the
//! target payload in the megakernel. Those crates reach a binary through
//! whatever depends on them, and a transitive dependency that references no
//! symbol in a crate lets the linker drop its object file, registrations
//! included. A reader would then see a registry with a level missing and no way
//! to tell that from a level that was never registered.
//!
//! Each source here is referenced by calling a real function in it, and the
//! floor is per source: a linked crate whose rows are absent from the registry
//! is a link failure this module reports rather than a shorter registry every
//! reader agrees with.

use std::sync::LazyLock;

use vyre_foundation::optimizer::level_contract::{
    registered_level_stages, stages_registered_here, LevelStage,
};
use vyre_spec::IrLevel;

/// Every crate that registers a level stage.
///
/// This is the list a crate joins when it takes ownership of a level's subject;
/// `registry_link_rules` reads the tree to prove the list is the whole set.
pub const DECLARED_SOURCES: &[&str] = &["vyre-foundation", "vyre-lower", "vyre-megakernel"];

/// One crate linked into this build, with the levels it registers stages for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LevelStageSource {
    /// Crate that owns the stages.
    pub crate_name: &'static str,
    /// Levels it registers a stage for.
    pub levels: Vec<IrLevel>,
}

/// Calling each crate's own accessor is what links it. A constant naming the
/// level would inline at the use site and link nothing.
static SOURCES: LazyLock<Vec<LevelStageSource>> = LazyLock::new(|| {
    vec![
        LevelStageSource {
            crate_name: "vyre-foundation",
            levels: stages_registered_here().to_vec(),
        },
        LevelStageSource {
            crate_name: "vyre-lower",
            levels: vec![vyre_lower::registered_level_stage()],
        },
        LevelStageSource {
            crate_name: "vyre-megakernel",
            levels: vec![vyre_megakernel::registered_level_stage()],
        },
    ]
});

/// Every crate linked into this build, with the levels it registers.
#[must_use]
pub fn linked_level_stage_sources() -> &'static [LevelStageSource] {
    &SOURCES
}

/// The level-stage registry, with every source crate referenced.
///
/// # Panics
///
/// Panics when a linked source registered no stage for a level it declares,
/// which means its object file was dropped at link time and every reader would
/// judge a registry with that level missing.
#[must_use]
pub fn live_level_stages() -> Vec<&'static (dyn LevelStage + Sync)> {
    let stages = registered_level_stages();
    assert_linked_sources_reached_registry(&stages);
    stages
}

/// Every level a linked source declares must be in the registry.
fn assert_linked_sources_reached_registry(stages: &[&'static (dyn LevelStage + Sync)]) {
    for source in linked_level_stage_sources() {
        for level in &source.levels {
            assert!(
                stages.iter().any(|stage| stage.level() == *level),
                "Fix: `{}` is linked and declares a stage for `{}`, but the registry has none. \
                 Its object file was dropped at link time; reference a real symbol in it.",
                source.crate_name,
                level.name()
            );
        }
    }
}
