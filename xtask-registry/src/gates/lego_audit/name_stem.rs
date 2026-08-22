//! Check 9: a leaf-stem family is namespaced, merged, or acknowledged.
//!
//! Four operations whose leaf names share a stem are a wall of near-synonyms to
//! anyone searching for the stem. The family is either explicit in the id, or
//! merged, or recorded in [`KNOWN_STEM_FAMILIES`] with the reason a rename is
//! not available.

use super::*;

pub(super) const STEM_COLLISION_MIN: usize = 4;

/// Stems whose family is acknowledged rather than renamed.
///
/// A row belongs here when the family is real and cannot be moved into a
/// namespace segment without renaming an op id that is already registered.
/// `dominator` is that case: `dominator_frontier` and `dominator_tree` predate
/// the two phase operations the fixpoint now composes, and `dominator::tree`
/// would rename both.
///
/// `check_0_every_exemption_is_live` holds each row to a stem that would be
/// reported without it, so a row outliving its family fails instead of reading
/// as a reviewed decision. The `opt` row was already dead when that rule landed.
pub(super) const KNOWN_STEM_FAMILIES: [&str; 13] = [
    "and",
    "ast",
    "attention",
    "csr",
    "dominator",
    "i4x8",
    "int4",
    "linear",
    "matmul",
    "python312",
    "quest",
    // Not one family: `graph::tensor_flow_*` is dataflow analysis over an AST,
    // `math::tensor_network_pair_contract` and `math::tensor_train_decompose`
    // are tensor-network algebra. They share the English word and nothing else,
    // so a `tensor::` segment would name a family that does not exist and would
    // rename four registered ids to do it.
    "tensor",
    "workgroup",
];

pub(super) fn is_known_stem_family(stem: &str) -> bool {
    KNOWN_STEM_FAMILIES.contains(&stem)
}

/// Stems the collision rule would report if the allowlist were empty.
///
/// A stem qualifies when at least `STEM_COLLISION_MIN` ops share it and they do
/// not already live under a namespace segment of that name, which is the family
/// being explicit by construction. Both the rule and the liveness check read
/// this, so an allowlist row is judged against the condition it suppresses
/// rather than against a second copy of it.
pub(super) fn colliding_stems(ops: &[OpInfo]) -> BTreeMap<String, Vec<String>> {
    let mut buckets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for op in ops {
        if is_internal_phase_op(&op.id) {
            continue;
        }
        let leaf = op.id.rsplit("::").next().unwrap_or(&op.id);
        let stem = leaf_stem(leaf);
        if stem.is_empty() {
            continue;
        }
        buckets
            .entry(stem.to_string())
            .or_default()
            .push(op.id.clone());
    }
    buckets.retain(|stem, ids| {
        ids.len() >= STEM_COLLISION_MIN
            && !ids
                .iter()
                .all(|id| id.contains(&format!("::{stem}::")) || id.ends_with(&format!("::{stem}")))
    });
    buckets
}

pub(super) fn check_9_name_stem_collision(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!(
        "Name-stem collision (≥ {STEM_COLLISION_MIN} ops sharing a leaf-prefix stem)"
    ));
    let mut flagged = 0usize;
    for (stem, ids) in colliding_stems(ops) {
        if is_known_stem_family(&stem) {
            continue;
        }
        report.find(violation(format!("  ⚠ {} ops share leaf-stem `{stem}`: {}. Fix: namespace the family (e.g. `{stem}::tiled`, `{stem}::strassen`), merge near-duplicates, or add a stem allowlist entry.",
            ids.len(),
            ids.join(", "))));
        flagged += 1;
    }
    if flagged == 0 {
        report.note(format!(
            "  ✓ no leaf-stem collisions ≥ {STEM_COLLISION_MIN}"
        ));
    }
    flagged
}

/// Reduce a leaf identifier to its discoverability stem: drop the
/// trailing `_<suffix>` segment so `matmul`, `matmul_tiled`,
/// `matmul_strassen`, `matmul_one_level` all map to `matmul`.
pub(super) fn leaf_stem(leaf: &str) -> &str {
    match leaf.find('_') {
        Some(idx) => &leaf[..idx],
        None => leaf,
    }
}

// ============================================================
// Check 10: unreviewed shape pair  -  catches false negatives of check 1.
// ============================================================
//
// Check 1 fires when bigram-cosine ≥ 0.88. False negatives slip when
// two ops share the same operand-type tuple AND the same fingerprint
// prefix (the first ~16 bytes of the IR-shape fingerprint, which
// captures the entry node-kind sequence). These are the "same
// problem, slightly reordered" duplicates that bigram cosine misses.
//
// WHY the score reads only past the prefix: the bucket key already
// fixes those bytes identical for every pair in the bucket, so scoring
// them again measures the key and reports similarity the check itself
// created. Two ops whose entries agree and whose remainders diverge
// scored above the threshold on the strength of the agreement that put
// them in one bucket. The remainder is the only evidence the key did
// not already spend, so the remainder is what the score reads, and a
// body that ends inside the key window carries no such evidence and is
// not compared at all.

#[cfg(test)]
mod tests {
    use super::*;

    /// This contract test keeps discoverability stems stable across multi-suffix operation names.
    #[test]
    fn leaf_stem_drops_first_underscore_suffix() {
        assert_eq!(leaf_stem("matmul"), "matmul");
        assert_eq!(leaf_stem("matmul_tiled"), "matmul");
        assert_eq!(leaf_stem("matmul_strassen_one_level"), "matmul");
        assert_eq!(leaf_stem("fft_radix2"), "fft");
        assert_eq!(leaf_stem(""), "");
    }

    /// This discoverability test preserves explicit acknowledgement of intentional operation families.
    #[test]
    fn known_stem_families_are_explicit() {
        assert!(is_known_stem_family("matmul"));
        assert!(is_known_stem_family("int4"));
        assert!(!is_known_stem_family("unreviewed"));
    }
}
