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
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
};

use super::arena_cursor::ArenaCursor;
use super::encode::EncodeError;
use super::expr_arena::{encode_expr_arena, ExprArenaEncoding};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

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
}

/// Errors surfaced by `gpu_cse_canonicals`.
#[derive(Debug)]
pub enum CseError {
    /// Encoder did not accept the input shape.
    Encode(EncodeError),
    /// Semantic execution or canonical output decoding failed.
    Semantic(SemanticExecutionError),
}

impl std::fmt::Display for CseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_cse encode error: {err:?}"),
            Self::Semantic(err) => write!(f, "gpu_cse semantic execution error: {err}"),
        }
    }
}

impl std::error::Error for CseError {}

/// Workgroup size used by both CSE kernels. Matches the rest of the
/// optimizer keystone for cache coherence on per-Expr passes.
pub const WORKGROUP_X: u32 = 256;

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
    ) -> Result<Self, SemanticExecutionError> {
        let count = pair_count as usize;
        let expected_words = count.checked_mul(2).ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} compact canonical pair count overflows usize: {pair_count}."
            ))
        })?;
        if pair_words.len() != expected_words {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} compact canonical expected {expected_words} pair word(s) for {pair_count} pair(s), got {}.",
                pair_words.len()
            )));
        }

        let mut overrides = FxHashMap::default();
        overrides.try_reserve(count).map_err(|error| {
            SemanticExecutionError::Backend(format!(
                "Fix: reserve {context} compact canonical map for {count} pair(s): {error}."
            ))
        })?;
        for pair in pair_words.chunks_exact(2) {
            let id = pair[0];
            let canonical = pair[1];
            if id >= expr_count || canonical >= expr_count {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: {context} compact canonical pair ({id}, {canonical}) exceeds expr_count {expr_count}."
                )));
            }
            if canonical > id {
                return Err(SemanticExecutionError::InvalidRequest(format!(
                    "Fix: {context} compact canonical pair ({id}, {canonical}) is not monotonic; canonical ids must be <= source ids."
                )));
            }
            if canonical == id {
                continue;
            }
            if let Some(previous) = overrides.insert(id, canonical) {
                return Err(SemanticExecutionError::InvalidRequest(format!(
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
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<(ExprArenaEncoding, Vec<u32>), CseError> {
    let arena = encode_expr_arena(program).map_err(CseError::Encode)?;
    let n = arena.expr_count;
    if n == 0 {
        return Ok((arena, Vec::new()));
    }
    let mut scratch = CseKernelScratch::default();
    let mut canonical = Vec::with_capacity(n as usize);
    run_cse_kernels_with_scratch_into(&arena, executor, policy, &mut scratch, &mut canonical)
        .map_err(CseError::Semantic)?;
    Ok((arena, canonical))
}

#[cfg(test)]
fn run_cse_kernels_into(
    arena: &ExprArenaEncoding,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    canonical: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = CseKernelScratch::default();
    run_cse_kernels_with_scratch_into(arena, executor, policy, &mut scratch, canonical)
}

fn run_cse_kernels_with_scratch_into(
    arena: &ExprArenaEncoding,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    scratch: &mut CseKernelScratch,
    canonical: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let n = arena.expr_count;
    let words = n as usize;

    scratch.max_depth[0] = arena.max_depth;
    ensure_input_slots(&mut scratch.hash_inputs, 6);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[3], &arena.arg2);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[4], &arena.depths);
    write_u32_slice_le_bytes(&mut scratch.hash_inputs[5], &scratch.max_depth);
    let mut hash_execution = vyre_megakernel::execute_single_program(
        executor,
        "cse-structural-hash",
        build_structural_hash_program(n, arena.max_depth.saturating_add(1).max(1)),
        &scratch.hash_inputs,
        policy,
    )?;
    if hash_execution.outputs.len() != 1 {
        return Err(SemanticExecutionError::Backend(format!(
            "cse structural-hash semantic execution expected one output, got {}",
            hash_execution.outputs.len()
        )));
    }
    let hash_bytes = hash_execution.outputs.remove(0);
    decode_u32_output_exact(&hash_bytes, words, "gpu_cse hash", &mut scratch.hash_words)
        .map_err(|error| SemanticExecutionError::Backend(format!("cse hash output: {error}")))?;

    ensure_input_slots(&mut scratch.canonical_inputs, 5);
    scratch.canonical_inputs[0].clear();
    scratch.canonical_inputs[0].extend_from_slice(&hash_bytes);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[1], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[2], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[3], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.canonical_inputs[4], &arena.arg2);
    let mut canonical_execution = vyre_megakernel::execute_single_program(
        executor,
        "cse-canonical-id",
        build_canonical_id_program(n),
        &scratch.canonical_inputs,
        policy,
    )?;
    if canonical_execution.outputs.len() != 1 {
        return Err(SemanticExecutionError::Backend(format!(
            "cse canonical-id semantic execution expected one canonical output, got {}",
            canonical_execution.outputs.len()
        )));
    }
    decode_u32_output_exact(
        &canonical_execution.outputs.remove(0),
        words,
        "gpu_cse canonical",
        canonical,
    )
    .map_err(|error| SemanticExecutionError::Backend(format!("cse canonical output: {error}")))
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
