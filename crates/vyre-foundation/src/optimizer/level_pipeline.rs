//! One pass pipeline per IR level.
//!
//! The scheduler owns one order over every registered pass, derived from
//! declared requirements and causal invalidation. That order says nothing about
//! which IR level each pass acts at, so a pass that rewrites a selected
//! schedule and a pass that rewrites logical arithmetic are neighbours in one
//! flat list. Two defects follow. A logical rewrite scheduled after a schedule
//! rewrite reads a program that already carries physical constructs, and its
//! preconditions were stated about a program that did not. And nothing states
//! the pipeline of a level, so a level's verifier, canonical form, and analysis
//! set have no set of passes to be the verifier of.
//!
//! This module partitions the scheduled order by the level each pass's
//! [`RewriteContract`](crate::optimizer::rewrite_contract::RewriteContract)
//! declares. The partition is derived, never listed: a pass appears in the
//! pipeline of the level its contract names, and a pass with no contract
//! appears in none, which is what `pass_invariants` reports.

use vyre_spec::IrLevel;

use super::derived_order::derive_registered_pass_order;
use super::rewrite_contract::contract_for_pass;
use super::OptimizerError;

/// The passes one IR level owns, in scheduled order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelPipeline {
    level: IrLevel,
    passes: Vec<&'static str>,
}

impl LevelPipeline {
    /// Level this pipeline rewrites.
    #[must_use]
    pub fn level(&self) -> IrLevel {
        self.level
    }

    /// Pass names in the order the scheduler runs them.
    #[must_use]
    pub fn passes(&self) -> &[&'static str] {
        &self.passes
    }

    /// Whether no registered pass declares this level.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

/// One pipeline per declared IR level, ordered from whole program to target
/// payload.
///
/// A level no registered pass declares still gets a row, because the levels are
/// the compiler's stages whether or not this crate ships a rewrite for one.
///
/// # Errors
///
/// Returns the scheduler's error when the registered pass set cannot be
/// ordered.
pub fn level_pipelines() -> Result<Vec<LevelPipeline>, OptimizerError> {
    let order = derive_registered_pass_order()?;
    let mut pipelines: Vec<LevelPipeline> = IrLevel::all()
        .iter()
        .map(|&level| LevelPipeline {
            level,
            passes: Vec::new(),
        })
        .collect();
    for node in order.nodes() {
        let Some(contract) = contract_for_pass(node.name) else {
            continue;
        };
        if let Some(pipeline) = pipelines
            .iter_mut()
            .find(|pipeline| pipeline.level == contract.level)
        {
            pipeline.passes.push(node.name);
        }
    }
    Ok(pipelines)
}

/// The level `pass` declares, when it declares a contract.
#[must_use]
pub fn level_of_pass(pass: &str) -> Option<IrLevel> {
    contract_for_pass(pass).map(|contract| contract.level)
}

/// One pass ordered before a pass of an earlier level.
///
/// `later` is scheduled after `earlier` and acts at a lower level, so the
/// earlier pass reads constructs a later stage introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelInversion {
    /// Pass scheduled first, acting at the higher level.
    pub earlier: &'static str,
    /// Level `earlier` declares.
    pub earlier_level: IrLevel,
    /// Pass scheduled after it, acting at the lower level.
    pub later: &'static str,
    /// Level `later` declares.
    pub later_level: IrLevel,
}

/// Every place the scheduled order runs a level before a level that precedes
/// it.
///
/// # Errors
///
/// Returns the scheduler's error when the registered pass set cannot be
/// ordered.
pub fn level_inversions() -> Result<Vec<LevelInversion>, OptimizerError> {
    let order = derive_registered_pass_order()?;
    let levelled: Vec<(&'static str, IrLevel)> = order
        .nodes()
        .iter()
        .filter_map(|node| level_of_pass(node.name).map(|level| (node.name, level)))
        .collect();
    Ok(inversions_in_order(&levelled))
}

/// Every place `order` runs a level before a level that precedes it.
///
/// `order` is a run order paired with the level each pass declares. The scan
/// carries the deepest level reached so far, so a pass shallower than anything
/// already run is reported against the deepest pass that preceded it rather
/// than against its immediate neighbour: the constructs it must not have read
/// were introduced by that pass.
#[must_use]
pub fn inversions_in_order(order: &[(&'static str, IrLevel)]) -> Vec<LevelInversion> {
    let mut inversions = Vec::new();
    let mut deepest: Option<(&'static str, IrLevel)> = None;
    for &(pass, level) in order {
        match deepest {
            Some((earlier, earlier_level)) if level < earlier_level => {
                inversions.push(LevelInversion {
                    earlier,
                    earlier_level,
                    later: pass,
                    later_level: level,
                });
            }
            Some((_, earlier_level)) if level > earlier_level => {
                deepest = Some((pass, level));
            }
            Some(_) => {}
            None => deepest = Some((pass, level)),
        }
    }
    inversions
}
