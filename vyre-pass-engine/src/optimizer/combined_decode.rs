//! Combined arena-pass delta application.
//!
//! Walks the input Program in the same DFS post-order as the
//! ExprArena encoder, applying the canonicalize swap_mask, const-fold
//! foldable+value, and pattern-match rewrite_action in priority order:
//!
//! 1. **const-fold** wins: if `foldable[id] == 1`, replace the Expr
//!    with `LitU32(value[id])`.
//! 2. **pattern-match** next: apply the `rewrite_action` from the
//!    pattern bank (replace with left/right child or LitU32(0)).
//! 3. **canonicalize** last: if it's a BinOp and `swap_mask[id] == 1`,
//!    swap operands.
//!
//! These precedence rules are shared by the semantic pass decoders.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_megakernel::SemanticExecutionError;

use super::pattern_match_via_encoded::rewrite_action as ra;

/// Lookup contract for arena-pass deltas.
pub trait ArenaDeltaLookup {
    /// Canonicalization swap flag for `id`.
    fn swap_mask(&self, id: usize) -> u32;

    /// Const-fold flag for `id`.
    fn foldable(&self, id: usize) -> u32;

    /// Const-fold value for `id`.
    fn value(&self, id: usize) -> u32;

    /// Pattern-match rewrite action for `id`.
    fn rewrite_action(&self, id: usize) -> u32;
}

struct DenseArenaDeltas<'a> {
    swap_mask: &'a [u32],
    foldable: &'a [u32],
    value: &'a [u32],
    rewrite_action: &'a [u32],
}

/// Fixed-size compressed arena deltas: bitsets for boolean planes,
/// dense u32 arrays for payload/action planes.
pub struct BitsetArenaDeltas<'a> {
    swap_bits: &'a [u32],
    fold_bits: &'a [u32],
    value: &'a [u32],
    rewrite_action: &'a [u32],
}

impl<'a> BitsetArenaDeltas<'a> {
    /// Build a borrowed compressed-delta view.
    #[must_use]
    pub fn new(
        swap_bits: &'a [u32],
        fold_bits: &'a [u32],
        value: &'a [u32],
        rewrite_action: &'a [u32],
    ) -> Self {
        Self {
            swap_bits,
            fold_bits,
            value,
            rewrite_action,
        }
    }

    fn bit(words: &[u32], id: usize) -> u32 {
        words
            .get(id / 32)
            .map(|word| (word >> (id % 32)) & 1)
            .unwrap_or(0)
    }
}

impl ArenaDeltaLookup for BitsetArenaDeltas<'_> {
    fn swap_mask(&self, id: usize) -> u32 {
        Self::bit(self.swap_bits, id)
    }

    fn foldable(&self, id: usize) -> u32 {
        Self::bit(self.fold_bits, id)
    }

    fn value(&self, id: usize) -> u32 {
        self.value.get(id).copied().unwrap_or(0)
    }

    fn rewrite_action(&self, id: usize) -> u32 {
        self.rewrite_action.get(id).copied().unwrap_or(ra::NONE)
    }
}

impl ArenaDeltaLookup for DenseArenaDeltas<'_> {
    fn swap_mask(&self, id: usize) -> u32 {
        self.swap_mask.get(id).copied().unwrap_or(0)
    }

    fn foldable(&self, id: usize) -> u32 {
        self.foldable.get(id).copied().unwrap_or(0)
    }

    fn value(&self, id: usize) -> u32 {
        self.value.get(id).copied().unwrap_or(0)
    }

    fn rewrite_action(&self, id: usize) -> u32 {
        self.rewrite_action.get(id).copied().unwrap_or(ra::NONE)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ArenaDeltaRecord {
    swap_mask: u32,
    foldable: u32,
    value: u32,
    rewrite_action: u32,
}

/// Sparse arena deltas decoded from resident device compaction.
#[derive(Debug, Clone, Default)]
pub struct SparseArenaDeltas {
    expr_count: u32,
    overrides: FxHashMap<u32, ArenaDeltaRecord>,
}

impl SparseArenaDeltas {
    /// Decode compacted records emitted by
    /// [`build_resident_delta_compact_program`].
    pub fn from_compacted_record_words(
        expr_count: u32,
        record_count: u32,
        record_words: &[u32],
        context: &str,
    ) -> Result<Self, SemanticExecutionError> {
        let count = record_count as usize;
        let expected_words = count.checked_mul(5).ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} compact arena record count overflows usize: {record_count}."
            ))
        })?;
        if record_words.len() != expected_words {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} compact arena expected {expected_words} record word(s) for {record_count} record(s), got {}.",
                record_words.len()
            )));
        }

        let mut overrides = FxHashMap::default();
        overrides.try_reserve(count).map_err(|error| {
            SemanticExecutionError::Backend(format!(
                "Fix: reserve {context} compact arena map for {count} record(s): {error}."
            ))
        })?;
        for record in record_words.chunks_exact(5) {
            let id = record[0];
            if id >= expr_count {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: {context} compact arena record id {id} exceeds expr_count {expr_count}."
                )));
            }
            let delta = ArenaDeltaRecord {
                swap_mask: record[1],
                foldable: record[2],
                value: record[3],
                rewrite_action: record[4],
            };
            if delta.swap_mask == 0 && delta.foldable == 0 && delta.rewrite_action == ra::NONE {
                continue;
            }
            if overrides.insert(id, delta).is_some() {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: {context} compact arena emitted duplicate expr id {id}."
                )));
            }
        }

        Ok(Self {
            expr_count,
            overrides,
        })
    }

    /// Number of non-identity arena delta records.
    #[must_use]
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    fn delta(&self, id: usize) -> Option<ArenaDeltaRecord> {
        let id_u32 = u32::try_from(id).ok()?;
        if id_u32 >= self.expr_count {
            return None;
        }
        self.overrides.get(&id_u32).copied()
    }
}

impl ArenaDeltaLookup for SparseArenaDeltas {
    fn swap_mask(&self, id: usize) -> u32 {
        self.delta(id).map(|delta| delta.swap_mask).unwrap_or(0)
    }

    fn foldable(&self, id: usize) -> u32 {
        self.delta(id).map(|delta| delta.foldable).unwrap_or(0)
    }

    fn value(&self, id: usize) -> u32 {
        self.delta(id).map(|delta| delta.value).unwrap_or(0)
    }

    fn rewrite_action(&self, id: usize) -> u32 {
        self.delta(id)
            .map(|delta| delta.rewrite_action)
            .unwrap_or(ra::NONE)
    }
}

/// Build the resident optimizer compaction Program.
///
/// Buffer layout:
///   0: swap_mask (RO)
///   1: foldable (RO)
///   2: value (RO)
///   3: rewrite_action (RO)
///   4: canonical (RO)
///   5: arena_delta_count (RW), word 0 = record count
///   6: arena_delta_records (RW), records are
///      `(expr_id, swap, foldable, value, rewrite_action)`
///   7: canonical_delta_count (RW), word 0 = pair count
///   8: canonical_delta_pairs (RW), records are
///      `(expr_id, canonical_id)`
#[must_use]
pub fn build_resident_delta_compact_program(expr_count: u32) -> Program {
    let arena_delta_words = expr_count.saturating_mul(5).max(1);
    let canonical_delta_words = expr_count.saturating_mul(2).max(1);
    let buffers = vec![
        BufferDecl::storage("swap_mask", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("foldable", 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("value", 2, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("rewrite_action", 3, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("canonical", 4, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage(
            "arena_delta_count",
            5,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
        BufferDecl::storage(
            "arena_delta_records",
            6,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(arena_delta_words),
        BufferDecl::storage(
            "canonical_delta_count",
            7,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
        BufferDecl::storage(
            "canonical_delta_pairs",
            8,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(canonical_delta_words),
    ];
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("swap", Expr::load("swap_mask", Expr::var("i"))),
                Node::let_bind("fold", Expr::load("foldable", Expr::var("i"))),
                Node::let_bind("val", Expr::load("value", Expr::var("i"))),
                Node::let_bind("action", Expr::load("rewrite_action", Expr::var("i"))),
                Node::if_then(
                    Expr::or(
                        Expr::or(
                            Expr::ne(Expr::var("swap"), Expr::u32(0)),
                            Expr::ne(Expr::var("fold"), Expr::u32(0)),
                        ),
                        Expr::ne(Expr::var("action"), Expr::u32(ra::NONE)),
                    ),
                    vec![
                        Node::let_bind(
                            "arena_slot",
                            Expr::atomic_add("arena_delta_count", Expr::u32(0), Expr::u32(1)),
                        ),
                        Node::let_bind(
                            "arena_base",
                            Expr::mul(Expr::var("arena_slot"), Expr::u32(5)),
                        ),
                        Node::store(
                            "arena_delta_records",
                            Expr::var("arena_base"),
                            Expr::var("i"),
                        ),
                        Node::store(
                            "arena_delta_records",
                            Expr::add(Expr::var("arena_base"), Expr::u32(1)),
                            Expr::var("swap"),
                        ),
                        Node::store(
                            "arena_delta_records",
                            Expr::add(Expr::var("arena_base"), Expr::u32(2)),
                            Expr::var("fold"),
                        ),
                        Node::store(
                            "arena_delta_records",
                            Expr::add(Expr::var("arena_base"), Expr::u32(3)),
                            Expr::var("val"),
                        ),
                        Node::store(
                            "arena_delta_records",
                            Expr::add(Expr::var("arena_base"), Expr::u32(4)),
                            Expr::var("action"),
                        ),
                    ],
                ),
                Node::let_bind("canonical_id", Expr::load("canonical", Expr::var("i"))),
                Node::if_then(
                    Expr::ne(Expr::var("canonical_id"), Expr::var("i")),
                    vec![
                        Node::let_bind(
                            "canonical_slot",
                            Expr::atomic_add("canonical_delta_count", Expr::u32(0), Expr::u32(1)),
                        ),
                        Node::let_bind(
                            "canonical_base",
                            Expr::mul(Expr::var("canonical_slot"), Expr::u32(2)),
                        ),
                        Node::store(
                            "canonical_delta_pairs",
                            Expr::var("canonical_base"),
                            Expr::var("i"),
                        ),
                        Node::store(
                            "canonical_delta_pairs",
                            Expr::add(Expr::var("canonical_base"), Expr::u32(1)),
                            Expr::var("canonical_id"),
                        ),
                    ],
                ),
            ],
        ),
    ];

    Program::wrapped(buffers, [256, 1, 1], body)
}

/// Build a fixed-size resident delta packer.
///
/// This compresses the two boolean delta planes (`swap_mask` and
/// `foldable`) into u32 bitsets so the release path can read one
/// bounded range set with no extra host fence. Dense value/action and
/// canonical planes are read directly from their existing resident
/// buffers by the caller.
#[must_use]
pub fn build_resident_delta_bitset_pack_program(expr_count: u32) -> Program {
    let bit_words = expr_count.div_ceil(32).max(1);
    let buffers = vec![
        BufferDecl::storage("swap_mask", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("foldable", 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("swap_bits", 2, BufferAccess::ReadWrite, DataType::U32)
            .with_count(bit_words),
        BufferDecl::storage("fold_bits", 3, BufferAccess::ReadWrite, DataType::U32)
            .with_count(bit_words),
    ];
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("word", Expr::div(Expr::var("i"), Expr::u32(32))),
                Node::let_bind(
                    "bit",
                    Expr::shl(Expr::u32(1), Expr::bitand(Expr::var("i"), Expr::u32(31))),
                ),
                Node::if_then(
                    Expr::ne(Expr::load("swap_mask", Expr::var("i")), Expr::u32(0)),
                    vec![Node::let_bind(
                        "swap_old",
                        Expr::atomic_or("swap_bits", Expr::var("word"), Expr::var("bit")),
                    )],
                ),
                Node::if_then(
                    Expr::ne(Expr::load("foldable", Expr::var("i")), Expr::u32(0)),
                    vec![Node::let_bind(
                        "fold_old",
                        Expr::atomic_or("fold_bits", Expr::var("word"), Expr::var("bit")),
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(buffers, [256, 1, 1], body)
}

/// Apply the combined per-Expr deltas to `program`, producing the
/// post-arena-pass Program. DCE is run separately on the result.
pub fn apply_combined_arena_deltas(
    program: &Program,
    swap_mask: &[u32],
    foldable: &[u32],
    value: &[u32],
    rewrite_action: &[u32],
) -> Program {
    let deltas = DenseArenaDeltas {
        swap_mask,
        foldable,
        value,
        rewrite_action,
    };
    apply_combined_arena_deltas_with_lookup(program, &deltas)
}

/// Sparse/dense-agnostic variant of [`apply_combined_arena_deltas`].
pub fn apply_combined_arena_deltas_with_lookup<D: ArenaDeltaLookup + ?Sized>(
    program: &Program,
    deltas: &D,
) -> Program {
    super::rewrite_walk::rewrite_program_with_expr_rewriter(program, |expr, counter| {
        super::rewrite_walk::rewrite_simple_expr_postorder(expr, counter, &mut |rebuilt, id| {
            arena_delta_decision(rebuilt, id, deltas)
        })
    })
}

/// Apply compressed bitset arena deltas.
pub fn apply_combined_arena_deltas_bitsets(
    program: &Program,
    swap_bits: &[u32],
    fold_bits: &[u32],
    value: &[u32],
    rewrite_action: &[u32],
) -> Program {
    let deltas = BitsetArenaDeltas::new(swap_bits, fold_bits, value, rewrite_action);
    apply_combined_arena_deltas_with_lookup(program, &deltas)
}

/// Combined rewrite decision for the expression rebuilt at arena id `id`.
///
/// Three arena passes are decoded in one walk, in the priority order this
/// module documents: const-fold, then the pattern-match rewrite action, then
/// the canonicalize operand swap.
fn arena_delta_decision<D: ArenaDeltaLookup + ?Sized>(rebuilt: Expr, id: u32, deltas: &D) -> Expr {
    let id = id as usize;
    match rebuilt {
        Expr::BinOp { op, left, right } => {
            if deltas.foldable(id) == 1 {
                let raw = deltas.value(id);
                // A comparison BinOp is semantically Bool, so emit `LitBool`:
                // dead-branch and the other type-aware passes downstream read
                // the literal's shape, not just its bits.
                let bool_result = matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                );
                return if bool_result {
                    Expr::LitBool(raw != 0)
                } else {
                    Expr::LitU32(raw)
                };
            }
            match deltas.rewrite_action(id) {
                ra::REPLACE_WITH_LEFT => return *left,
                ra::REPLACE_WITH_RIGHT => return *right,
                ra::REPLACE_WITH_LIT_ZERO => return Expr::LitU32(0),
                ra::REPLACE_WITH_LIT_TRUE => return Expr::LitBool(true),
                ra::REPLACE_WITH_LIT_FALSE => return Expr::LitBool(false),
                ra::REPLACE_WITH_LEFT_INNER_LEFT => {
                    if let Expr::BinOp { left: inner, .. } = left.as_ref() {
                        return inner.as_ref().clone();
                    }
                }
                ra::REPLACE_WITH_LEFT_INNER_RIGHT => {
                    if let Expr::BinOp { right: inner, .. } = left.as_ref() {
                        return inner.as_ref().clone();
                    }
                }
                _ => {}
            }
            if deltas.swap_mask(id) == 1 {
                Expr::BinOp {
                    op,
                    left: right,
                    right: left,
                }
            } else {
                Expr::BinOp { op, left, right }
            }
        }
        Expr::UnOp { op, operand } => {
            if deltas.foldable(id) == 1 {
                return Expr::LitU32(deltas.value(id));
            }
            // `REPLACE_WITH_GRAND_OPERAND` fires for `~~x`, `--x`, `!!x`; the
            // surviving Expr is one level below the rebuilt operand.
            if deltas.rewrite_action(id) == ra::REPLACE_WITH_GRAND_OPERAND {
                if let Expr::UnOp { operand: inner, .. } = operand.as_ref() {
                    return inner.as_ref().clone();
                }
            }
            Expr::UnOp { op, operand }
        }
        other => {
            // Loads, selects, and FMAs are never folded, matched, or swapped by
            // the V1 rule sets; they keep their rebuilt children.
            if matches!(
                other,
                Expr::Load { .. } | Expr::Select { .. } | Expr::Fma { .. }
            ) {
                other
            } else {
                decide_leaf(&other, id, deltas)
            }
        }
    }
}

fn decide_leaf<D: ArenaDeltaLookup + ?Sized>(expr: &Expr, id: usize, deltas: &D) -> Expr {
    if deltas.foldable(id) == 1 {
        match expr {
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) => expr.clone(),
            _ => Expr::LitU32(deltas.value(id)),
        }
    } else {
        expr.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_delta_compact_program_carries_arena_and_canonical_outputs() {
        let p = build_resident_delta_compact_program(8);
        assert!(p.buffers().iter().any(|b| b.name() == "arena_delta_count"));
        assert!(p
            .buffers()
            .iter()
            .any(|b| b.name() == "arena_delta_records"));
        assert!(p
            .buffers()
            .iter()
            .any(|b| b.name() == "canonical_delta_count"));
        assert!(p
            .buffers()
            .iter()
            .any(|b| b.name() == "canonical_delta_pairs"));
    }

    #[test]
    fn resident_delta_bitset_pack_program_carries_boolean_planes() {
        let p = build_resident_delta_bitset_pack_program(65);
        assert!(p.buffers().iter().any(|b| b.name() == "swap_bits"));
        assert!(p.buffers().iter().any(|b| b.name() == "fold_bits"));
    }

    #[test]
    fn bitset_arena_deltas_decode_boolean_planes() {
        let deltas = BitsetArenaDeltas::new(&[0b10, 0b1], &[0b100], &[0, 7, 9], &[0, 0, 3]);
        assert_eq!(deltas.swap_mask(1), 1);
        assert_eq!(deltas.swap_mask(32), 1);
        assert_eq!(deltas.swap_mask(2), 0);
        assert_eq!(deltas.foldable(2), 1);
        assert_eq!(deltas.value(2), 9);
        assert_eq!(deltas.rewrite_action(2), 3);
    }

    #[test]
    fn sparse_arena_deltas_default_identity_and_override_changed_records() {
        let deltas = SparseArenaDeltas::from_compacted_record_words(
            8,
            2,
            &[3, 1, 0, 0, ra::NONE, 5, 0, 1, 99, ra::REPLACE_WITH_LIT_ZERO],
            "test sparse arena",
        )
        .expect("Fix: valid compact arena records decode");
        assert_eq!(deltas.override_count(), 2);
        assert_eq!(deltas.swap_mask(0), 0);
        assert_eq!(deltas.swap_mask(3), 1);
        assert_eq!(deltas.foldable(5), 1);
        assert_eq!(deltas.value(5), 99);
        assert_eq!(deltas.rewrite_action(5), ra::REPLACE_WITH_LIT_ZERO);
    }

    #[test]
    fn sparse_arena_deltas_reject_malformed_record_count() {
        let err = SparseArenaDeltas::from_compacted_record_words(
            8,
            2,
            &[3, 1, 0, 0, ra::NONE],
            "test sparse arena",
        )
        .expect_err("compact arena record count must match record words exactly");
        assert!(
            matches!(err, SemanticExecutionError::InvalidRequest(_)),
            "unexpected error: {err:?}"
        );
    }
}
