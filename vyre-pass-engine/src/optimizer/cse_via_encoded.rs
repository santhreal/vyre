//! GPU-native common-subexpression elimination on the encoded arena.
//!
//! Two passes operating on the canonical 5-buffer arena:
//!
//! 1. **structural_hash**  -  level-parallel bottom-up. Each Expr's hash
//!    is `mix(kind, payload, child_hashes...)` so that two arena rows
//!    representing the same syntactic Expr collapse to the same hash.
//!    Runs in a single fused kernel with workgroup-scope barriers
//!    between levels (same single-workgroup pattern as the fused
//!    const-fold). One dispatch.
//!
//! 2. **canonical_id**  -  for each hash bucket, the smallest expr id
//!    with that hash wins. Implemented as a length-`2*expr_count`
//!    open-addressed direct map with atomic-min on the value slot.
//!    Linear probing on hash collision. Capacity > 2× ensures load
//!    factor ≤ 0.5 and bounded probe length. One dispatch.
//!
//! After both passes, `canonical[i]` gives the smallest expr id
//! structurally equivalent to `i`. Identity (`canonical[i] == i`)
//! means `i` is its own canonical; otherwise `i` is a duplicate of
//! `canonical[i]`. The IR rewrite that consumes `canonical[]` is in
//! `apply_cse_canonicals`.
//!
//! Hash function: a Fowler–Noll–Vo–style mix over kind / op / child
//! hashes / payload. 32-bit; collision probability over 5k-expr
//! arenas is ~0.3% per arena. Future versions can promote to 64-bit
//! for stronger guarantees; the architecture here doesn't change.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{Expr, Ident, Node, Program};
use vyre_foundation::transform::rewrite_walk::{rewrite_program_entry, rewrite_scope, NodeRewrite};

use vyre_libs::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};

use super::arena_cursor::ArenaCursor;
use super::encode::EncodeError;
use super::expr_arena::{encode_expr_arena, ExprArenaEncoding};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub use super::cse_cross_scope::{apply_cross_scope_cse, apply_cross_scope_cse_with_lookup};
pub use super::cse_programs::{
    build_canonical_delta_compact_program, build_canonical_id_program,
    build_structural_hash_program,
};
#[derive(Debug, Default)]
struct CseKernelScratch {
    hash_inputs: Vec<Vec<u8>>,
    canonical_inputs: Vec<Vec<u8>>,
    max_depth: [u32; 1],
    hash_words: Vec<u32>,
    table_init_words: Vec<u32>,
}

/// Errors surfaced by `gpu_cse_canonicals`.
#[derive(Debug)]
pub enum CseError {
    /// Encoder did not accept the input shape.
    Encode(EncodeError),
    /// Dispatcher rejected or failed.
    Dispatch(DispatchError),
}

impl std::fmt::Display for CseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_cse encode error: {err:?}"),
            Self::Dispatch(err) => write!(f, "gpu_cse dispatch error: {err}"),
        }
    }
}

impl std::error::Error for CseError {}

/// Workgroup size used by both CSE kernels. Matches the rest of the
/// optimizer keystone for cache coherence on per-Expr passes.
pub const WORKGROUP_X: u32 = 256;

/// Capacity multiplier for the canonical-id direct-map. Must stay
/// strictly above `1` so the table's load factor stays bounded; `2`
/// keeps probe length to a small constant in expectation.
pub const CANONICAL_TABLE_MULT: u32 = 2;

/// Lookup contract for CSE canonical ids.
///
/// Dense GPU CSE returns `canonical[id]` for every arena id. Resident
/// pipelines can instead read back only non-identity pairs from a
/// device-side compaction kernel; consumers should not care which
/// representation produced the lookup.
pub trait CanonicalLookup {
    /// True when no non-identity canonical mappings exist.
    fn is_empty(&self) -> bool;

    /// Return the canonical id for `id`, defaulting to identity.
    fn canonical_of(&self, id: u32) -> u32;
}

impl CanonicalLookup for [u32] {
    fn is_empty(&self) -> bool {
        <[u32]>::is_empty(self)
    }

    fn canonical_of(&self, id: u32) -> u32 {
        self.get(id as usize).copied().unwrap_or(id)
    }
}

/// Return whether two node-owned top-level expressions share one canonical id.
///
/// Both let-level and cross-scope CSE can only rewrite complete expressions
/// attached directly to nodes. Duplicate inner expressions alone cannot make
/// either pass productive, so this bounded preflight avoids cloning and walking
/// a large program when the GPU canonical table contains no actionable pair.
pub(super) fn has_repeated_top_level_canonical<C: CanonicalLookup + ?Sized>(
    arena: &ExprArenaEncoding,
    canonical: &C,
) -> bool {
    if canonical.is_empty() || arena.expr_count == 0 {
        return false;
    }
    let mut seen = vec![false; arena.expr_count as usize];
    for expr_id in arena.node_top_level_exprs.iter().flatten().copied() {
        let canonical_id = canonical.canonical_of(expr_id);
        let Some(slot) = seen.get_mut(canonical_id as usize) else {
            // A malformed lookup must not suppress the normal rewrite path.
            return true;
        };
        if *slot {
            return true;
        }
        *slot = true;
    }
    false
}

/// Sparse canonical map decoded from `(expr_id, canonical_id)` pairs.
#[derive(Debug, Clone, Default)]
pub struct SparseCanonicalMap {
    expr_count: u32,
    overrides: FxHashMap<u32, u32>,
}

impl SparseCanonicalMap {
    /// Decode compacted pair words emitted by
    /// [`build_canonical_delta_compact_program`].
    pub fn from_compacted_pair_words(
        expr_count: u32,
        pair_count: u32,
        pair_words: &[u32],
        context: &str,
    ) -> Result<Self, DispatchError> {
        let count = pair_count as usize;
        let expected_words = count.checked_mul(2).ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: {context} compact canonical pair count overflows usize: {pair_count}."
            ))
        })?;
        if pair_words.len() != expected_words {
            return Err(DispatchError::BadInputs(format!(
                "Fix: {context} compact canonical expected {expected_words} pair word(s) for {pair_count} pair(s), got {}.",
                pair_words.len()
            )));
        }

        let mut overrides = FxHashMap::default();
        overrides.try_reserve(count).map_err(|error| {
            DispatchError::BackendError(format!(
                "Fix: reserve {context} compact canonical map for {count} pair(s): {error}."
            ))
        })?;
        for pair in pair_words.chunks_exact(2) {
            let id = pair[0];
            let canonical = pair[1];
            if id >= expr_count || canonical >= expr_count {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: {context} compact canonical pair ({id}, {canonical}) exceeds expr_count {expr_count}."
                )));
            }
            if canonical > id {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: {context} compact canonical pair ({id}, {canonical}) is not monotonic; canonical ids must be <= source ids."
                )));
            }
            if canonical == id {
                continue;
            }
            if let Some(previous) = overrides.insert(id, canonical) {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: {context} compact canonical emitted duplicate source id {id} with values {previous} and {canonical}."
                )));
            }
        }

        Ok(Self {
            expr_count,
            overrides,
        })
    }

    /// Number of non-identity canonical overrides.
    #[must_use]
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }
}

impl CanonicalLookup for SparseCanonicalMap {
    fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    fn canonical_of(&self, id: u32) -> u32 {
        if id >= self.expr_count {
            return id;
        }
        self.overrides.get(&id).copied().unwrap_or(id)
    }
}

/// Run CSE analysis. Returns a `canonical` vector where `canonical[i]`
/// is the smallest expr id structurally equivalent to `i`. Identity
/// (`canonical[i] == i`) means `i` is its own canonical. Use the
/// returned vector with `apply_cse_canonicals` to rewrite the
/// Program.
pub fn gpu_cse_canonicals(
    program: &Program,
    dispatcher: &dyn ProgramDispatcher,
) -> Result<(ExprArenaEncoding, Vec<u32>), CseError> {
    let arena = encode_expr_arena(program).map_err(CseError::Encode)?;
    let n = arena.expr_count;
    if n == 0 {
        return Ok((arena, Vec::new()));
    }
    let mut scratch = CseKernelScratch::default();
    let mut canonical = Vec::with_capacity(n as usize);
    run_cse_kernels_with_scratch_into(&arena, dispatcher, &mut scratch, &mut canonical)
        .map_err(CseError::Dispatch)?;
    Ok((arena, canonical))
}

#[cfg(test)]
fn run_cse_kernels_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    canonical: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = CseKernelScratch::default();
    run_cse_kernels_with_scratch_into(arena, dispatcher, &mut scratch, canonical)
}

fn run_cse_kernels_with_scratch_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    scratch: &mut CseKernelScratch,
    canonical: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let n = arena.expr_count;
    let words = n as usize;
    let state_bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: gpu_cse state byte count overflows usize for expr_count={n}."
            ))
        })?;

    // ---- Pass A: structural hash ----------------------------------
    let hash_program = build_structural_hash_program(n, arena.max_depth.saturating_add(1).max(1));
    scratch.max_depth[0] = arena.max_depth;
    ensure_input_slots(&mut scratch.hash_inputs, 7);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[3], &arena.arg2);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[4], &arena.depths);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[5], &scratch.max_depth);
    write_zero_bytes(&mut scratch.hash_inputs[6], state_bytes);
    let hash_outputs = dispatcher.dispatch(&hash_program, &scratch.hash_inputs, Some([1, 1, 1]))?;
    if hash_outputs.len() != 1 {
        return Err(DispatchError::BackendError(format!(
            "Fix: gpu_cse hash dispatch expected exactly one hash output, got {}.",
            hash_outputs.len()
        )));
    }
    decode_u32_output_exact(
        &hash_outputs[0],
        words,
        "gpu_cse hash",
        &mut scratch.hash_words,
    )?;

    // ---- Pass B: canonical-id direct-map --------------------------
    let capacity = (n.saturating_mul(CANONICAL_TABLE_MULT)).max(2);
    let canonical_program = build_canonical_id_program(n, capacity);
    // Initial state for table_canonical: u32::MAX (empty marker).
    scratch.table_init_words.clear();
    scratch.table_init_words.resize(capacity as usize, u32::MAX);
    // 7 inputs: hash, canonical (RW scratch), table_canonical (RW dummy),
    // arena_kinds, arena_arg0, arena_arg1, arena_arg2. The four arena
    // buffers supply the structural tuple comparison that makes the
    // hash-equality pre-filter sound; without them a 32-bit hash collision
    // would silently merge non-equivalent exprs.
    ensure_input_slots(&mut scratch.canonical_inputs, 7);
    scratch.canonical_inputs[0].clear();
    scratch.canonical_inputs[0].extend_from_slice(&hash_outputs[0]);
    write_zero_bytes(&mut scratch.canonical_inputs[1], state_bytes);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[2], &scratch.table_init_words);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[3], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[4], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[5], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[6], &arena.arg2);
    let canonical_outputs = dispatcher.dispatch(
        &canonical_program,
        &scratch.canonical_inputs,
        // Grid covers expr_count threads (n_workgroups = ceil(n/256))
        Some([(n + WORKGROUP_X - 1) / WORKGROUP_X, 1, 1]),
    )?;
    if canonical_outputs.is_empty() {
        return Err(DispatchError::BackendError(format!(
            "Fix: gpu_cse canonical dispatch expected at least one canonical output, got {}.",
            canonical_outputs.len()
        )));
    }
    decode_u32_output_exact(&canonical_outputs[0], words, "gpu_cse canonical", canonical)
}

/// Apply `canonical[i]` to rewrite `program`. Replaces every
/// `Node::Let` whose value-Expr is a CSE duplicate with
/// `Expr::Var(orig_name)`, where `orig_name` is the first binding in
/// the same scope that produced the canonical expression.
///
/// Delegates to [`apply_cse_let_dedupe_with_lookup`], which implements
/// the correct let-level rewrite using `arena.node_top_level_exprs` to
/// correlate node walk order with arena expr ids. The two functions
/// must produce identical results for the let-level case; use this
/// entry point when you have a dense `canonical` slice.
pub use apply_cse_let_dedupe as apply_cse_canonicals;

/// Apply a let-level CSE rewrite: when an entire `Node::Let { name,
/// value: V }` has a value-Expr structurally equivalent to an earlier
/// Let's value in the SAME scope, replace `V` with `Expr::Var(orig)`.
/// This is the safe-by-construction subset of CSE rewrite  -  no
/// cross-scope hoisting needed.
///
/// Walks the program in the same DFS order the arena encoder uses.
/// Tracks a per-scope `expr_id → let_name` map; entering a new scope
/// (If/Loop/Block branches) pushes a fresh map so duplicates only
/// dedupe against same-scope siblings.
pub fn apply_cse_let_dedupe(
    program: &Program,
    arena: &ExprArenaEncoding,
    canonical: &[u32],
) -> Program {
    apply_cse_let_dedupe_with_lookup(program, arena, canonical)
}

/// Sparse/dense-agnostic variant of [`apply_cse_let_dedupe`].
pub fn apply_cse_let_dedupe_with_lookup<C: CanonicalLookup + ?Sized>(
    program: &Program,
    arena: &ExprArenaEncoding,
    canonical: &C,
) -> Program {
    if !has_repeated_top_level_canonical(arena, canonical) {
        return program.clone();
    }
    let mut walker = LetDedupeWalker {
        cursor: ArenaCursor::at_first_real_node(&arena.node_top_level_exprs),
        canonical,
        scope: FxHashMap::default(),
        pending: None,
    };
    rewrite_program_entry(program, |body| walker.rewrite_scope(body))
}

struct LetDedupeWalker<'a, C: CanonicalLookup + ?Sized> {
    cursor: ArenaCursor<'a>,
    canonical: &'a C,
    /// Per-scope map: canonical id of a Let's value -> that Let's name. Two
    /// Let nodes in the SAME scope whose values are CSE-equivalent share a
    /// canonical id, and the later one is rewritten to read the earlier name.
    scope: FxHashMap<u32, Ident>,
    /// The substitution the current node's single operand is owed, decided in
    /// [`NodeRewrite::enter`] where the Let's name is in hand.
    pending: Option<Expr>,
}

impl<C: CanonicalLookup + ?Sized> LetDedupeWalker<'_, C> {
    fn rewrite_scope(&mut self, body: &[Node]) -> Vec<Node> {
        // Entering a scope starts a fresh map, so a duplicate only dedupes
        // against a sibling binding that is still live where it is read.
        let enclosing = std::mem::take(&mut self.scope);
        let out = rewrite_scope(body, self);
        self.scope = enclosing;
        out
    }
}

impl<C: CanonicalLookup + ?Sized> NodeRewrite for LetDedupeWalker<'_, C> {
    /// The dedupe decision needs the Let's name, which no single position
    /// carries, so it is taken here and applied when the value is offered.
    ///
    /// `Assign` is deliberately not deduplicated: it reassigns an existing
    /// binding, so substituting the earlier name would change what the program
    /// observes at run time.
    fn enter(&mut self, node: &Node) {
        let top_ids = self.cursor.take_node();
        self.pending = None;
        let Node::Let { name, .. } = node else {
            return;
        };
        let Some(top_id) = top_ids.first().copied() else {
            return;
        };
        let canon = self.canonical.canonical_of(top_id);
        if canon != top_id {
            if let Some(original) = self.scope.get(&canon) {
                // Duplicate. Keep the original canonical mapping so a third
                // occurrence reads the same binding.
                self.pending = Some(Expr::var(original.clone()));
                return;
            }
        }
        // First occurrence of this canonical value in this scope.
        self.scope.insert(canon, name.clone());
    }

    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        self.pending.take()
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        Some(self.rewrite_scope(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use vyre_libs::dispatch_buffers::u32_slice_to_le_bytes;

    struct CseDispatcher {
        outputs: RefCell<Vec<Vec<Vec<u8>>>>,
    }

    impl ProgramDispatcher for CseDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            Ok(self.outputs.borrow_mut().remove(0))
        }
    }

    use super::super::arena_kernel::single_lit_u32_arena as one_expr_arena;

    #[test]
    fn structural_hash_program_compiles_to_program() {
        let p = build_structural_hash_program(8, 4);
        assert!(p.buffers().iter().any(|b| b.name() == "hash"));
        assert!(p.buffers().iter().any(|b| b.name() == "max_depth_buf"));
    }

    #[test]
    fn canonical_id_program_carries_table_buffer() {
        let p = build_canonical_id_program(8, 16);
        assert!(p.buffers().iter().any(|b| b.name() == "canonical"));
        assert!(p.buffers().iter().any(|b| b.name() == "table_canonical"));
        // Structural tuple buffers must be present so the kernel can
        // confirm hash-equal exprs are actually structurally identical.
        assert!(
            p.buffers().iter().any(|b| b.name() == "arena_kinds"),
            "canonical-id program must declare arena_kinds for structural tuple check"
        );
        assert!(
            p.buffers().iter().any(|b| b.name() == "arena_arg0"),
            "canonical-id program must declare arena_arg0 for structural tuple check"
        );
        assert!(
            p.buffers().iter().any(|b| b.name() == "arena_arg1"),
            "canonical-id program must declare arena_arg1 for structural tuple check"
        );
        assert!(
            p.buffers().iter().any(|b| b.name() == "arena_arg2"),
            "canonical-id program must declare arena_arg2 for structural tuple check"
        );
    }

    #[test]
    fn canonical_delta_compact_program_carries_sparse_output_buffer() {
        let p = build_canonical_delta_compact_program(8);
        assert!(p.buffers().iter().any(|b| b.name() == "canonical"));
        assert!(p.buffers().iter().any(|b| b.name() == "canonical_delta"));
    }

    #[test]
    fn sparse_canonical_map_defaults_identity_and_overrides_duplicates() {
        let map = SparseCanonicalMap::from_compacted_pair_words(
            8,
            2,
            &[3, 1, 7, 2],
            "test sparse canonical",
        )
        .expect("Fix: valid compact canonical pairs decode");
        assert_eq!(map.override_count(), 2);
        assert_eq!(map.canonical_of(0), 0);
        assert_eq!(map.canonical_of(3), 1);
        assert_eq!(map.canonical_of(7), 2);
    }

    #[test]
    fn top_level_canonical_preflight_ignores_inner_only_duplicates() {
        let arena = ExprArenaEncoding {
            expr_count: 4,
            node_top_level_exprs: vec![Vec::new(), vec![1], vec![3]],
            ..ExprArenaEncoding::default()
        };
        assert!(
            !has_repeated_top_level_canonical(&arena, &[0, 1, 1, 3][..]),
            "an inner duplicate cannot make a node-level CSE rewrite productive"
        );
        assert!(
            has_repeated_top_level_canonical(&arena, &[0, 1, 2, 1][..]),
            "equivalent node-owned expressions must keep the CSE rewrite enabled"
        );
    }

    #[test]
    fn sparse_canonical_map_rejects_malformed_pair_count() {
        let err =
            SparseCanonicalMap::from_compacted_pair_words(8, 2, &[3, 1], "test sparse canonical")
                .expect_err("compact canonical pair count must match pair words exactly");
        assert!(
            matches!(err, DispatchError::BadInputs(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn cse_kernels_decode_exact_canonical_into_reused_buffer() {
        let dispatcher = CseDispatcher {
            outputs: RefCell::new(vec![
                vec![u32_slice_to_le_bytes(&[123])],
                vec![u32_slice_to_le_bytes(&[0])],
            ]),
        };
        let mut canonical = Vec::with_capacity(4);
        let ptr = canonical.as_ptr();
        run_cse_kernels_into(&one_expr_arena(), &dispatcher, &mut canonical)
            .expect("Fix: dispatch succeeds");
        assert_eq!(canonical, vec![0]);
        assert_eq!(canonical.as_ptr(), ptr);
    }

    #[test]
    fn cse_kernels_with_scratch_reuse_dispatch_decode_and_output_storage() {
        let dispatcher = CseDispatcher {
            outputs: RefCell::new(vec![
                vec![u32_slice_to_le_bytes(&[123])],
                vec![u32_slice_to_le_bytes(&[0])],
                vec![u32_slice_to_le_bytes(&[123])],
                vec![u32_slice_to_le_bytes(&[0])],
            ]),
        };
        let arena = one_expr_arena();
        let mut scratch = CseKernelScratch::default();
        let mut canonical = Vec::with_capacity(1);

        run_cse_kernels_with_scratch_into(&arena, &dispatcher, &mut scratch, &mut canonical)
            .expect("Fix: dispatch succeeds");

        let hash_input_capacities = scratch
            .hash_inputs
            .iter()
            .map(Vec::capacity)
            .collect::<Vec<_>>();
        let canonical_input_capacities = scratch
            .canonical_inputs
            .iter()
            .map(Vec::capacity)
            .collect::<Vec<_>>();
        let hash_words_capacity = scratch.hash_words.capacity();
        let table_capacity = scratch.table_init_words.capacity();
        let canonical_capacity = canonical.capacity();

        run_cse_kernels_with_scratch_into(&arena, &dispatcher, &mut scratch, &mut canonical)
            .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch
                .hash_inputs
                .iter()
                .map(Vec::capacity)
                .collect::<Vec<_>>(),
            hash_input_capacities
        );
        assert_eq!(
            scratch
                .canonical_inputs
                .iter()
                .map(Vec::capacity)
                .collect::<Vec<_>>(),
            canonical_input_capacities
        );
        assert_eq!(scratch.hash_words.capacity(), hash_words_capacity);
        assert_eq!(scratch.table_init_words.capacity(), table_capacity);
        assert_eq!(canonical.capacity(), canonical_capacity);
        assert_eq!(canonical, vec![0]);
    }

    #[test]
    fn cse_rejects_extra_hash_outputs() {
        let dispatcher = CseDispatcher {
            outputs: RefCell::new(vec![vec![
                u32_slice_to_le_bytes(&[123]),
                u32_slice_to_le_bytes(&[0]),
            ]]),
        };
        let mut canonical = Vec::new();
        let err = run_cse_kernels_into(&one_expr_arena(), &dispatcher, &mut canonical)
            .expect_err("extra hash outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn cse_rejects_trailing_canonical_bytes() {
        let dispatcher = CseDispatcher {
            outputs: RefCell::new(vec![
                vec![u32_slice_to_le_bytes(&[123])],
                vec![vec![0, 0, 0, 0, 1]],
            ]),
        };
        let mut canonical = Vec::new();
        let err = run_cse_kernels_into(&one_expr_arena(), &dispatcher, &mut canonical)
            .expect_err("trailing canonical bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    /// P0 regression: the canonical dispatch must receive all 7 input buffers
    /// (hash + canonical + table_canonical + 4 arena structural buffers).
    /// Before the fix, only 3 inputs were wired: the structural tuple buffers
    /// were absent, so the hash-only pre-filter was the sole equivalence
    /// predicate and a 32-bit collision would silently merge distinct exprs.
    #[test]
    fn canonical_dispatch_receives_seven_inputs_including_arena_structural_buffers() {
        use std::cell::Cell;
        struct InputCountDispatcher {
            canonical_input_count: Cell<usize>,
            call: Cell<usize>,
        }
        impl ProgramDispatcher for InputCountDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                let call = self.call.get();
                self.call.set(call + 1);
                if call == 1 {
                    // Second dispatch = canonical-id program.
                    self.canonical_input_count.set(inputs.len());
                }
                // Return one zero-word output (expr_count = 1).
                Ok(vec![u32_slice_to_le_bytes(&[0])])
            }
        }
        let arena = one_expr_arena();
        let dispatcher = InputCountDispatcher {
            canonical_input_count: Cell::new(0),
            call: Cell::new(0),
        };
        let mut canonical = Vec::new();
        run_cse_kernels_into(&arena, &dispatcher, &mut canonical)
            .expect("Fix: cse kernels dispatch succeeds");
        assert_eq!(
            dispatcher.canonical_input_count.get(),
            7,
            "canonical-id dispatch must receive 7 inputs: hash, canonical (RW), \
             table_canonical (RW dummy), arena_kinds, arena_arg0, arena_arg1, arena_arg2; \
             before the fix only 3 inputs were wired and hash collisions silently merged \
             non-equivalent exprs"
        );
    }

    /// P1 regression: `apply_cse_canonicals` must actually rewrite duplicate
    /// `Let` bindings. A program with two identical `LitU32(42)` bindings and
    /// a canonical map that points the second expr to the first should produce
    /// `let b = Var("a")`, not `let b = LitU32(42)`.
    #[test]
    fn apply_cse_canonicals_rewrites_duplicate_let_to_var() {
        use vyre_foundation::ir::{Expr, Ident, Node, Program};
        // Program:
        //   let a = LitU32(42)   // expr 0 in arena  → canonical[0] = 0 (self)
        //   let b = LitU32(42)   // expr 1 in arena  → canonical[1] = 0 (dup of a)
        let entry = vec![
            Node::let_bind("a", Expr::u32(42)),
            Node::let_bind("b", Expr::u32(42)),
        ];
        let prog = Program::wrapped(Vec::new(), [1, 1, 1], entry);
        let arena = encode_expr_arena(&prog).expect("Fix: simple program encodes");
        // expr 0 = LitU32(42) for 'a', expr 1 = LitU32(42) for 'b'.
        // Canonical: b's expr (id=1) is a dup of a's expr (id=0).
        assert_eq!(arena.expr_count, 2, "expected 2 exprs in arena");
        let canonical = vec![0u32, 0u32]; // canonical[1] = 0
        let rewritten = apply_cse_canonicals(&prog, &arena, &canonical);
        // Expect: let b = Var("a")
        let entry_nodes: Vec<Node> = match rewritten.entry() {
            [Node::Region { body, .. }] => body.as_ref().to_vec(),
            other => other.to_vec(),
        };
        assert_eq!(entry_nodes.len(), 2, "program must still have 2 nodes");
        match &entry_nodes[1] {
            Node::Let { name, value } => {
                assert_eq!(name.as_ref(), "b", "second let must remain named 'b'");
                assert_eq!(
                    value,
                    &Expr::Var(Ident::new(std::sync::Arc::from("a"))),
                    "apply_cse_canonicals must rewrite let b = LitU32(42) to let b = Var(\"a\") \
                     when canonical[1] == 0 and the canonical expr is bound to 'a'; \
                     before the fix the function was a no-op stub that returned the original program"
                );
            }
            other => panic!("expected Node::Let for 'b', got {other:?}"),
        }
    }
}
