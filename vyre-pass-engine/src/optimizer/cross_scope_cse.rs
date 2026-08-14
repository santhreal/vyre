//! Cross-scope expression CSE  -  hoists common subexpressions to
//! shared `let` bindings.
//!
//! Complements `apply_cse_let_dedupe` (which dedupes only `let`-RHS
//! pairs in the same scope) by handling Exprs that appear directly
//! as the value of `Node::Store` / `Node::Assign` / `Node::If` cond
//! / `Node::Loop` bounds. When the same canonical-equivalent Expr
//! appears at 2+ such top-level positions in a single scope, we
//! introduce a fresh `let __cse_<n> = E;` at the scope's start and
//! rewrite each occurrence to `Var(__cse_<n>)`.
//!
//! Walker uses the arena's `node_top_level_exprs` to identify per-
//! Node arena ids  -  robust to upstream rewrites that change inner
//! Expr shape, since Node-level structure is preserved.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{Expr, Ident, Node, Program};
use vyre_foundation::transform::rewrite_walk::{self, NodeRewrite};

use super::arena_cursor::ArenaCursor;
use super::cse_via_encoded::{has_repeated_top_level_canonical, CanonicalLookup};
use super::expr_arena::ExprArenaEncoding;
use super::expr_no_atomic;

/// Apply same-scope expression CSE. For each scope, identifies
/// non-trivial top-level Exprs whose canonical id is shared and
/// hoists them to a single `let __cse_<n> = E;` at the scope's
/// start.
pub fn apply_cross_scope_cse(
    program: &Program,
    arena: &ExprArenaEncoding,
    canonical: &[u32],
) -> Program {
    apply_cross_scope_cse_with_lookup(program, arena, canonical)
}

/// Sparse/dense-agnostic variant of [`apply_cross_scope_cse`].
pub fn apply_cross_scope_cse_with_lookup<C: CanonicalLookup + ?Sized>(
    program: &Program,
    arena: &ExprArenaEncoding,
    canonical: &C,
) -> Program {
    if program.stats().node_count < 2 {
        return program.clone();
    }
    if !has_repeated_top_level_canonical(arena, canonical) {
        return program.clone();
    }
    let mut hoister = Hoister {
        cursor: ArenaCursor::at_first_real_node(&arena.node_top_level_exprs),
        canonical,
        next_let_id: 0,
    };
    super::rewrite_program_entry(program, |body| hoister.rewrite_scope(body))
}

/// One top-level Expr position within a single scope's direct
/// Nodes. The canonical id and a clone of the actual Expr.
struct Occurrence {
    canon: u32,
    expr: Expr,
}

struct Hoister<'a, C: CanonicalLookup + ?Sized> {
    cursor: ArenaCursor<'a>,
    canonical: &'a C,
    /// Monotonic suffix for fresh `__cse_N` names.
    next_let_id: u32,
}

impl<C: CanonicalLookup + ?Sized> Hoister<'_, C> {
    /// Hoist within one scope, then recurse into the nested scopes.
    ///
    /// Two passes over the same nodes, and they must agree exactly on which
    /// arena id each operand position carries: a position counted in pass 1
    /// and skipped in pass 2 leaves a hoisted `let` nobody reads, and the
    /// reverse substitutes a name that was never bound. Both drive
    /// [`rewrite_walk::rewrite_node`], so the agreement is structural rather
    /// than two hand-written matches kept in step by review.
    fn rewrite_scope(&mut self, body: &[Node]) -> Vec<Node> {
        let prefix_len = super::encode::reachable_prefix_len(body);
        let scope_start = self.cursor.position();

        let mut collect = CollectOccurrences {
            hoister: self,
            top_ids: Vec::new(),
            slot: 0,
            occurrences: Vec::new(),
        };
        for node in &body[..prefix_len] {
            rewrite_walk::rewrite_node(node, &mut collect);
        }
        let occurrences = collect.occurrences;

        // Identify hoist-worthy canonicals (count >= 2, non-trivial,
        // atomic-free).
        let mut counts: FxHashMap<u32, (u32, Expr)> = FxHashMap::default();
        let mut order: Vec<u32> = Vec::new();
        for occ in occurrences {
            counts
                .entry(occ.canon)
                .and_modify(|(c, _)| *c += 1)
                .or_insert_with(|| {
                    order.push(occ.canon);
                    (1, occ.expr)
                });
        }
        let mut plan: FxHashMap<u32, Ident> = FxHashMap::default();
        let mut hoist_lets: Vec<Node> = Vec::new();
        for canon in &order {
            let Some((count, expr)) = counts.get(canon).cloned() else {
                continue;
            };
            if count < 2 || !is_hoist_worthy(&expr) || !expr_no_atomic(&expr) {
                continue;
            }
            let name = self.fresh_name();
            hoist_lets.push(Node::let_bind(name.clone(), expr));
            plan.insert(*canon, name);
        }

        // Pass 2 restarts the cursor at the top of this scope so it sees the
        // identical id for every position pass 1 counted.
        self.cursor.rewind_to(scope_start);
        let mut out: Vec<Node> = Vec::with_capacity(prefix_len + hoist_lets.len());
        out.extend(hoist_lets);
        let mut substitute = SubstituteHoisted {
            hoister: self,
            plan: &plan,
            top_ids: Vec::new(),
            slot: 0,
        };
        for node in &body[..prefix_len] {
            out.push(
                rewrite_walk::rewrite_node(node, &mut substitute).unwrap_or_else(|| node.clone()),
            );
        }
        out
    }

    fn fresh_name(&mut self) -> Ident {
        let id = self.next_let_id;
        self.next_let_id += 1;
        Ident::new(Arc::from(format!("__cse_{id}")))
    }
}

/// Pass 1: record each direct operand's canonical id, and advance the cursor
/// past nested scopes without recording them. A nested scope hoists into its
/// own start, so its occurrences belong to its own plan.
struct CollectOccurrences<'h, 'a, C: CanonicalLookup + ?Sized> {
    hoister: &'h mut Hoister<'a, C>,
    top_ids: Vec<u32>,
    slot: usize,
    occurrences: Vec<Occurrence>,
}

impl<C: CanonicalLookup + ?Sized> NodeRewrite for CollectOccurrences<'_, '_, C> {
    fn enter(&mut self, _node: &Node) {
        self.top_ids = self.hoister.cursor.take_node();
        self.slot = 0;
    }

    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        let slot = self.slot;
        self.slot += 1;
        if let Some(arena_id) = self.top_ids.get(slot).copied() {
            self.occurrences.push(Occurrence {
                canon: self.hoister.canonical.canonical_of(arena_id),
                expr: expr.clone(),
            });
        }
        None
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        self.hoister.cursor.skip_body(body);
        None
    }
}

/// Pass 2: replace a planned position with the hoisted `Var`, and hoist the
/// nested scopes in turn.
struct SubstituteHoisted<'h, 'a, C: CanonicalLookup + ?Sized> {
    hoister: &'h mut Hoister<'a, C>,
    plan: &'h FxHashMap<u32, Ident>,
    top_ids: Vec<u32>,
    slot: usize,
}

impl<C: CanonicalLookup + ?Sized> NodeRewrite for SubstituteHoisted<'_, '_, C> {
    fn enter(&mut self, _node: &Node) {
        self.top_ids = self.hoister.cursor.take_node();
        self.slot = 0;
    }

    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        let slot = self.slot;
        self.slot += 1;
        let arena_id = self.top_ids.get(slot).copied()?;
        let canon = self.hoister.canonical.canonical_of(arena_id);
        self.plan.get(&canon).map(|name| Expr::var(name.clone()))
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        Some(self.hoister.rewrite_scope(body))
    }
}

/// Decide if an Expr is worth hoisting. Skip leaves  -  duplicating
/// those is cheaper than an extra Var indirection.
fn is_hoist_worthy(expr: &Expr) -> bool {
    !matches!(
        expr,
        Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF32(_)
            | Expr::LitBool(_)
            | Expr::Var(_)
            | Expr::BufLen { .. }
            | Expr::InvocationId { .. }
            | Expr::WorkgroupId { .. }
            | Expr::LocalId { .. }
            | Expr::SubgroupLocalId
            | Expr::SubgroupSize
    )
}
