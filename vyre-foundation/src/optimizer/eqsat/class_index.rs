//! Dense `EClass` index arithmetic and the checked reserve, dedup, and hash
//! helpers the `EGraph` tables are built from.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use super::{log_egraph_compat_error, EClassId, EGraphError, ENodeLang};

pub(super) fn eclass_id_from_index(index: usize) -> EClassId {
    match try_eclass_id_from_index(index) {
        Ok(id) => id,
        Err(error) => {
            log_egraph_compat_error("egraph class id conversion", &error);
            EClassId(0)
        }
    }
}

pub(super) fn try_eclass_id_from_index(index: usize) -> Result<EClassId, EGraphError> {
    u32::try_from(index)
        .map(EClassId)
        .map_err(|_| EGraphError::ClassIdOverflow { index })
}

pub(super) fn eclass_index(
    id: EClassId,
    len: usize,
    context: &'static str,
) -> Result<usize, EGraphError> {
    let index =
        usize::try_from(id.0).map_err(|_| EGraphError::ClassIdOutOfBounds { context, id, len })?;
    if index < len {
        Ok(index)
    } else {
        Err(EGraphError::ClassIdOutOfBounds { context, id, len })
    }
}

pub(super) fn reserve_vec_exact<T>(
    vec: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), EGraphError> {
    vec.try_reserve_exact(additional)
        .map_err(|source| EGraphError::Capacity {
            context,
            requested: additional,
            source: source.to_string(),
        })
}

pub(super) fn reserve_hashcons<L: Eq + Hash>(
    hashcons: &mut FxHashMap<L, EClassId>,
    additional: usize,
    context: &'static str,
) -> Result<(), EGraphError> {
    hashcons
        .try_reserve(additional)
        .map_err(|source| EGraphError::Capacity {
            context,
            requested: additional,
            source: source.to_string(),
        })
}

pub(super) fn dedup_enodes_by_hash<L: ENodeLang>(nodes: &mut Vec<L>) {
    if let Err(error) = try_dedup_enodes_by_hash(nodes) {
        log_egraph_compat_error("egraph dedup", &error);
    }
}

pub(super) fn try_dedup_enodes_by_hash<L: ENodeLang>(
    nodes: &mut Vec<L>,
) -> Result<(), EGraphError> {
    if nodes.len() <= 1 {
        return Ok(());
    }
    let mut keyed = Vec::new();
    reserve_vec_exact(&mut keyed, nodes.len(), "egraph dedup hash staging")?;
    keyed.extend(nodes.drain(..).map(|node| (stable_enode_hash(&node), node)));
    keyed.sort_unstable_by_key(|(hash, _)| *hash);
    let mut deduped: Vec<(u64, L)> = Vec::new();
    reserve_vec_exact(&mut deduped, keyed.len(), "egraph dedup output staging")?;
    for (hash, node) in keyed {
        let duplicate_in_hash_bucket = deduped
            .iter()
            .rev()
            .take_while(|(existing_hash, _)| *existing_hash == hash)
            .any(|(_, existing)| existing == &node);
        if !duplicate_in_hash_bucket {
            deduped.push((hash, node));
        }
    }
    reserve_vec_exact(nodes, deduped.len(), "egraph dedup node restoration")?;
    nodes.extend(deduped.into_iter().map(|(_, node)| node));
    Ok(())
}

pub(super) fn stable_enode_hash<L: ENodeLang>(node: &L) -> u64 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::super::EGraphError;
    use super::try_eclass_id_from_index;

    #[test]
    fn try_class_id_from_index_rejects_overflow() {
        if usize::BITS <= u32::BITS {
            return;
        }
        let overflow_index = (u32::MAX as usize) + 1;
        let err = try_eclass_id_from_index(overflow_index)
            .expect_err("overflowing class index must be rejected");
        assert_eq!(
            err,
            EGraphError::ClassIdOverflow {
                index: overflow_index
            }
        );
    }
}
