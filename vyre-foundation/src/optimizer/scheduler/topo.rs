//! Pass-graph topological scheduling: `schedule_passes` free fn +
//! `PassSchedulingError` + `next_ready_pass` helper.
//! Audit cleanup A21 (2026-04-30): split from monolithic scheduler.rs.

#![allow(unused_imports)]

use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hash};

use crate::allocation::{try_reserve_hash_map_to_capacity, try_reserve_vec_to_capacity};
use crate::optimizer::rewrite_contract::contract_for_pass;
use crate::optimizer::{PassMetadata, ProgramPassRegistration};

/// Describes errors that can occur during pass scheduling.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PassSchedulingError {
    /// A required pass is not in the set.
    #[error("optimizer pass `{pass}` requires unknown pass `{missing}`.")]
    UnknownRequire {
        /// The pass requiring the dependency.
        pass: &'static str,
        /// The missing dependency.
        missing: &'static str,
    },
    /// A cycle was detected in the pass dependencies.
    #[error("optimizer pass dependency cycle among {pass_ids:?}. Fix: {fix}")]
    Cycle {
        /// The passes involved in the cycle.
        pass_ids: Vec<&'static str>,
        /// A suggestion to break the cycle.
        fix: &'static str,
    },
    /// A pass with a duplicate ID was provided.
    #[error("duplicate pass id `{id}`.")]
    DuplicateId {
        /// The duplicated pass ID.
        id: &'static str,
    },
    /// A scheduled order placed a pass before one of its declared requirements.
    #[error("optimizer pass `{pass}` is scheduled before required pass `{requirement}`.")]
    OrderViolation {
        /// The pass requiring the dependency.
        pass: &'static str,
        /// The dependency that must appear first.
        requirement: &'static str,
    },
    /// Scheduler scratch allocation failed before graph traversal.
    #[error(
        "optimizer pass scheduler could not reserve {requested} {context} slot(s): {message}. Fix: reduce the pass set or schedule it in shards."
    )]
    StorageReserveFailed {
        /// Scratch vector or map being reserved.
        context: &'static str,
        /// Requested target capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

/// Computes a valid execution order for the given passes according to their requirements.
///
/// # Errors
///
/// Returns [`PassSchedulingError`] when required pass IDs are missing, pass IDs
/// are duplicated, or the dependency graph contains a cycle.
pub fn schedule_passes(
    passes: &[&'static ProgramPassRegistration],
) -> Result<Vec<&'static ProgramPassRegistration>, PassSchedulingError> {
    let mut metadata = Vec::new();
    reserve_vec_capacity(&mut metadata, passes.len(), "pass metadata")?;
    metadata.extend(passes.iter().map(|pass| pass.metadata));
    let order = schedule_pass_metadata_indices(&metadata)?;
    let mut scheduled = Vec::new();
    reserve_vec_capacity(&mut scheduled, order.len(), "scheduled pass output")?;
    scheduled.extend(order.into_iter().map(|index| passes[index]));
    Ok(scheduled)
}

pub(crate) fn schedule_pass_metadata_indices(
    passes: &[PassMetadata],
) -> Result<Vec<usize>, PassSchedulingError> {
    let n = passes.len();
    let mut by_id = FxHashMap::default();
    reserve_hash_map_capacity(&mut by_id, n, "pass id lookup")?;
    for (i, pass) in passes.iter().enumerate() {
        if by_id.insert(pass.name, i).is_some() {
            return Err(PassSchedulingError::DuplicateId { id: pass.name });
        }
    }

    let mut indegree = Vec::new();
    reserve_vec_capacity(&mut indegree, n, "pass indegree table")?;
    indegree.resize(n, 0usize);
    let mut dependents = Vec::new();
    reserve_vec_capacity(&mut dependents, n, "pass dependents table")?;
    dependents.resize_with(n, Vec::new);

    for (i, pass) in passes.iter().enumerate() {
        for required in pass.requires {
            if let Some(&req_i) = by_id.get(required) {
                if !dependents[req_i].contains(&i) {
                    dependents[req_i].push(i);
                    indegree[i] += 1;
                }
            } else {
                return Err(PassSchedulingError::UnknownRequire {
                    pass: pass.name,
                    missing: required,
                });
            }
        }
    }
    let ranks = level_ranks(passes);
    for children in &mut dependents {
        children.sort_unstable_by_key(|&child| (ranks[child], passes[child].name));
    }

    let mut initial_ready = Vec::new();
    reserve_vec_capacity(&mut initial_ready, n, "initial ready pass queue")?;
    initial_ready.extend(
        indegree
            .iter()
            .enumerate()
            .filter_map(|(id, &degree)| (degree == 0).then_some(id)),
    );
    initial_ready.sort_unstable_by_key(|&id| (ranks[id], passes[id].name));
    let mut ready = VecDeque::from(initial_ready);

    let mut ordered = Vec::new();
    reserve_vec_capacity(&mut ordered, n, "scheduled pass indices")?;
    while let Some(id) = ready.pop_front() {
        ordered.push(id);
        for &child in &dependents[id] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                let child_key = (ranks[child], passes[child].name);
                let pos = ready
                    .iter()
                    .position(|&existing| child_key < (ranks[existing], passes[existing].name))
                    .unwrap_or(ready.len());
                ready.insert(pos, child);
            }
        }
    }

    if ordered.len() != n {
        let mut pass_ids = Vec::new();
        reserve_vec_capacity(&mut pass_ids, n - ordered.len(), "cycle pass ids")?;
        pass_ids.extend(
            indegree
                .into_iter()
                .enumerate()
                .filter_map(|(id, degree)| (degree > 0).then_some(passes[id].name)),
        );
        pass_ids.sort_unstable();
        return Err(PassSchedulingError::Cycle {
            pass_ids,
            fix: "Break the cycle by removing one of these `requires` entries.",
        });
    }

    Ok(ordered)
}

/// Level rank per pass, so a ready pass at an earlier level is scheduled first.
///
/// The rank is the position of the level the pass's rewrite contract declares
/// in `IrLevel::all()`, which orders whole program before logical before
/// schedule before physical kernel. A pass that declares no contract ranks
/// last: nothing states the level it rewrites, so it cannot be placed among the
/// levels, and `pass_invariants` reports the missing contract.
///
/// Declared requirements still decide the order. This is the tie break between
/// passes that are ready at the same time, which was the pass name alone, so a
/// synchronization rewrite named early in the alphabet ran before every logical
/// rewrite.
fn level_ranks(passes: &[PassMetadata]) -> Vec<usize> {
    let levels = vyre_spec::IrLevel::all();
    passes
        .iter()
        .map(|pass| {
            contract_for_pass(pass.name)
                .and_then(|contract| levels.iter().position(|level| *level == contract.level))
                .unwrap_or(levels.len())
        })
        .collect()
}

pub(super) fn reserve_vec_capacity<T>(
    vec: &mut Vec<T>,
    requested: usize,
    context: &'static str,
) -> Result<(), PassSchedulingError> {
    try_reserve_vec_to_capacity(vec, requested).map_err(|source| {
        PassSchedulingError::StorageReserveFailed {
            context,
            requested,
            message: source.to_string(),
        }
    })
}

pub(super) fn reserve_hash_map_capacity<K, V, S>(
    map: &mut std::collections::HashMap<K, V, S>,
    requested: usize,
    context: &'static str,
) -> Result<(), PassSchedulingError>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    try_reserve_hash_map_to_capacity(map, requested).map_err(|source| {
        PassSchedulingError::StorageReserveFailed {
            context,
            requested,
            message: source.to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{level_ranks, schedule_pass_metadata_indices};
    use crate::optimizer::PassMetadata;

    /// A pass with no declared contract ranks after every level.
    ///
    /// Nothing states the level such a pass rewrites, so placing it among the
    /// levels would order it against a claim nobody made. Every registered pass
    /// declares a contract, which is why this is stated here rather than
    /// observed through the registry: `pass_invariants` reports the missing
    /// contract, and this fixes what the order does until it is declared.
    #[test]
    fn a_pass_with_no_contract_ranks_after_every_level() {
        let levels = vyre_spec::IrLevel::all().len();
        let unknown = PassMetadata::new("topo_test_pass_with_no_contract", &[], &[]);
        let declared = PassMetadata::new("canonicalize", &[], &[]);
        let ranks = level_ranks(&[unknown, declared]);
        assert_eq!(ranks[0], levels);
        assert!(
            ranks[1] < levels,
            "the canonicalize pass declares a level, so its rank is one of them"
        );
    }

    /// A ready pass at an earlier level is scheduled before a deeper one
    /// whatever the names say.
    #[test]
    fn the_tie_break_prefers_the_earlier_level() {
        // `atomic_minimize` declares the schedule level and sorts first by name;
        // `canonicalize` declares the logical level and sorts second.
        let order = schedule_pass_metadata_indices(&[
            PassMetadata::new("atomic_minimize", &[], &[]),
            PassMetadata::new("canonicalize", &[], &[]),
        ])
        .expect("two independent passes schedule");
        assert_eq!(
            order,
            vec![1, 0],
            "the logical pass runs first because its level precedes the schedule level"
        );
    }
}
