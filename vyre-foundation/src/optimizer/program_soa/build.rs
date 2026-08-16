//! Building the fact table: one preorder walk of the program tree.
//!
//! The walk fills every column in `facts.rs` and is the only writer of them.
//! A caller that needs the same facts twice takes the thread-local cache
//! rather than walking again.

use std::sync::OnceLock;

use super::facts::{ProgramFacts, RegionMeta};
use super::kind::{kind_mask, BufferRefKind, NodeIndex, NodeKind};
use crate::ir::{Expr, Ident, Node, Program};

thread_local! {
    /// Last (program-fingerprint, ProgramFacts) pair the current thread
    /// computed. ProgramFacts builds are deterministic in `program` and
    /// the scheduler runs sequentially against the SAME program for a
    /// burst of passes (analyze + transform per pass, multiple passes
    /// per iteration). A one-entry thread-local cache keyed by the
    /// program's stable fingerprint collapses 6+ redundant rebuilds
    /// per scheduler iteration into a single build.
    ///
    /// Rc rather than Arc  -  the cache slot only ever hands references
    /// back to the same thread that owns it, so we don't need cross-
    /// thread synchronization for the cached payload.
    static FACTS_CACHE: std::cell::RefCell<Option<([u8; 32], std::rc::Rc<ProgramFacts>)>> =
        const { std::cell::RefCell::new(None) };
}

impl ProgramFacts {
    /// Return a thread-local cached [`ProgramFacts`] for `program`,
    /// rebuilding only when the program's stable fingerprint differs
    /// from the last build on this thread.
    ///
    /// Use this in pass `analyze_impl` / `transform` paths instead of
    /// calling [`ProgramFacts::build`] directly: the scheduler hits
    /// the same `Program` repeatedly within one iteration (analyze
    /// then transform; multiple consecutive passes that all need
    /// facts) and the cache turns those repeats into refcount bumps.
    ///
    /// First-call cost is identical to `build`. Subsequent same-program
    /// calls on the same thread cost one `program.fingerprint()` (already
    /// OnceLock-cached) plus an Rc clone.
    #[must_use]
    pub fn build_cached(program: &Program) -> std::rc::Rc<ProgramFacts> {
        let fp = program.fingerprint();
        FACTS_CACHE.with(|cell| {
            let mut slot = cell.borrow_mut();
            if let Some((cached_fp, cached)) = slot.as_ref() {
                if cached_fp == &fp {
                    return std::rc::Rc::clone(cached);
                }
            }
            let facts = std::rc::Rc::new(ProgramFacts::build(program));
            *slot = Some((fp, std::rc::Rc::clone(&facts)));
            facts
        })
    }

    /// Walk the program's entry tree once in preorder and populate
    /// every column. The lookup indices are built lazily on the
    /// first call to `let_sites_of` / `var_read_sites_of` /
    /// `buffer_refs_of`.
    #[must_use]
    pub fn build(program: &Program) -> Self {
        match Self::try_build(program) {
            Ok(facts) => facts,
            Err(error) => {
                tracing::error!(
                    error = %error,
                    "ProgramFacts::build failed; use try_build on release paths to handle allocation pressure explicitly"
                );
                Self::default()
            }
        }
    }

    /// Fallible version of [`ProgramFacts::build`] for release paths that must
    /// surface allocation pressure instead of panicking during optimizer
    /// analysis.
    ///
    /// # Errors
    ///
    /// Returns an actionable message when a ProgramFacts column cannot reserve
    /// enough storage for the program's cached node/region counts.
    pub fn try_build(program: &Program) -> Result<Self, String> {
        // Pre-size the columnar Vec storage off the OnceLock-cached
        // node count so the build walk fills already-allocated
        // capacity instead of grow-by-doubling each column. The
        // counts (kinds, parent, lets, assigns, etc.) are bounded by
        // node_count; non-Let/Assign columns over-reserve, but the
        // single allocation is cheaper than 6+ doublings on a 1000-
        // node entry tree.
        let stats = program.stats();
        let node_count = stats.node_count;
        let mut facts = Self {
            kinds: Vec::default(),
            parent: Vec::default(),
            kinds_present: 0,
            lets: Vec::default(),
            assigns: Vec::default(),
            loop_vars: Vec::default(),
            var_reads: Vec::default(),
            buffer_refs: Vec::default(),
            regions: Vec::default(),
            let_index: OnceLock::new(),
            assign_index: OnceLock::new(),
            var_read_index: OnceLock::new(),
            buffer_index: OnceLock::new(),
            region_index_by_node: OnceLock::new(),
            region_index_by_generator: OnceLock::new(),
        };
        reserve_program_fact_columns(&mut facts, node_count, stats.region_count as usize)?;
        for node in program.entry() {
            walk_node(node, None, &mut facts);
        }
        Ok(facts)
    }
}

fn reserve_program_fact_columns(
    facts: &mut ProgramFacts,
    node_count: usize,
    region_count: usize,
) -> Result<(), String> {
    reserve_fact_vec(&mut facts.kinds, node_count, "kind column")?;
    reserve_fact_vec(&mut facts.parent, node_count, "parent column")?;
    reserve_fact_vec(&mut facts.lets, node_count / 4, "let column")?;
    reserve_fact_vec(&mut facts.assigns, node_count / 8, "assign column")?;
    reserve_fact_vec(&mut facts.loop_vars, node_count / 16, "loop-var column")?;
    reserve_fact_vec(&mut facts.var_reads, node_count, "var-read column")?;
    reserve_fact_vec(&mut facts.buffer_refs, node_count / 2, "buffer-ref column")?;
    reserve_fact_vec(&mut facts.regions, region_count, "region metadata column")?;
    Ok(())
}

fn reserve_fact_vec<T>(
    vec: &mut Vec<T>,
    target_capacity: usize,
    label: &'static str,
) -> Result<(), String> {
    crate::allocation::try_reserve_vec_to_capacity(vec, target_capacity).map_err(|source| {
        format!(
            "ProgramFacts {label} reservation failed for {target_capacity} item(s): {source}. Fix: shard the optimizer input or rebuild facts from a smaller program slice."
        )
    })
}

fn record_node(facts: &mut ProgramFacts, kind: NodeKind, parent: Option<NodeIndex>) -> NodeIndex {
    let idx = NodeIndex(u32::try_from(facts.kinds.len()).map_or(u32::MAX, |value| value));
    facts.kinds.push(kind);
    facts.parent.push(parent);
    // Set the kind-presence bit. `kind as u32` is the discriminant
    // (NodeKind has 16 variants, all fit in a u32). The optimizer
    // uses kinds_present for O(1) `has_kind` queries instead of
    // scanning the kinds column.
    facts.kinds_present |= kind_mask(kind);
    idx
}

#[expect(
    clippy::too_many_lines,
    reason = "SoA extraction keeps the Node variant-to-column mapping auditable in one walk"
)]
fn walk_node(node: &Node, parent: Option<NodeIndex>, facts: &mut ProgramFacts) {
    match node {
        Node::Let { name, value } => {
            let idx = record_node(facts, NodeKind::Let, parent);
            facts.lets.push((idx, name.duplicate_handle()));
            walk_expr(value, idx, facts);
        }
        Node::Assign { name, value } => {
            let idx = record_node(facts, NodeKind::Assign, parent);
            facts.assigns.push((idx, name.duplicate_handle()));
            walk_expr(value, idx, facts);
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            let idx = record_node(facts, NodeKind::Store, parent);
            facts
                .buffer_refs
                .push((idx, buffer.duplicate_handle(), BufferRefKind::Write));
            walk_expr(index, idx, facts);
            walk_expr(value, idx, facts);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let idx = record_node(facts, NodeKind::If, parent);
            walk_expr(cond, idx, facts);
            for n in then {
                walk_node(n, Some(idx), facts);
            }
            for n in otherwise {
                walk_node(n, Some(idx), facts);
            }
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let idx = record_node(facts, NodeKind::Loop, parent);
            facts.loop_vars.push((idx, var.duplicate_handle()));
            walk_expr(from, idx, facts);
            walk_expr(to, idx, facts);
            for n in body {
                walk_node(n, Some(idx), facts);
            }
        }
        Node::IndirectDispatch { count_buffer, .. } => {
            let idx = record_node(facts, NodeKind::IndirectDispatch, parent);
            facts.buffer_refs.push((
                idx,
                count_buffer.duplicate_handle(),
                BufferRefKind::IndirectCount,
            ));
        }
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            ..
        } => {
            let idx = record_node(facts, NodeKind::AsyncLoad, parent);
            facts
                .buffer_refs
                .push((idx, source.duplicate_handle(), BufferRefKind::AsyncSource));
            facts.buffer_refs.push((
                idx,
                destination.duplicate_handle(),
                BufferRefKind::AsyncDestination,
            ));
            walk_expr(offset, idx, facts);
            walk_expr(size, idx, facts);
        }
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            ..
        } => {
            let idx = record_node(facts, NodeKind::AsyncStore, parent);
            facts
                .buffer_refs
                .push((idx, source.duplicate_handle(), BufferRefKind::AsyncSource));
            facts
                .buffer_refs
                .push((idx, destination.duplicate_handle(), BufferRefKind::Write));
            walk_expr(offset, idx, facts);
            walk_expr(size, idx, facts);
        }
        Node::AsyncWait { .. } => {
            record_node(facts, NodeKind::AsyncWait, parent);
        }
        Node::Trap { address, .. } => {
            let idx = record_node(facts, NodeKind::Trap, parent);
            walk_expr(address, idx, facts);
        }
        Node::Resume { .. } => {
            record_node(facts, NodeKind::Resume, parent);
        }
        Node::Return => {
            record_node(facts, NodeKind::Return, parent);
        }
        Node::Barrier { .. } => {
            record_node(facts, NodeKind::Barrier, parent);
        }
        Node::Block(body) => {
            let idx = record_node(facts, NodeKind::Block, parent);
            for n in body {
                walk_node(n, Some(idx), facts);
            }
        }
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            let idx = record_node(facts, NodeKind::Region, parent);
            facts.regions.push(RegionMeta {
                node: idx,
                generator: generator.duplicate_handle(),
                source_region: source_region.as_ref().map(Ident::duplicate_handle),
            });
            for n in body.iter() {
                walk_node(n, Some(idx), facts);
            }
        }
        Node::AllReduce { buffer, .. } => {
            let idx = record_node(facts, NodeKind::AllReduce, parent);
            facts
                .buffer_refs
                .push((idx, buffer.duplicate_handle(), BufferRefKind::Write));
        }
        Node::AllGather { input, output, .. } => {
            let idx = record_node(facts, NodeKind::AllGather, parent);
            facts
                .buffer_refs
                .push((idx, input.duplicate_handle(), BufferRefKind::Read));
            facts
                .buffer_refs
                .push((idx, output.duplicate_handle(), BufferRefKind::Write));
        }
        Node::ReduceScatter { input, output, .. } => {
            let idx = record_node(facts, NodeKind::ReduceScatter, parent);
            facts
                .buffer_refs
                .push((idx, input.duplicate_handle(), BufferRefKind::Read));
            facts
                .buffer_refs
                .push((idx, output.duplicate_handle(), BufferRefKind::Write));
        }
        Node::Broadcast { buffer, .. } => {
            let idx = record_node(facts, NodeKind::Broadcast, parent);
            facts
                .buffer_refs
                .push((idx, buffer.duplicate_handle(), BufferRefKind::Write));
        }
        Node::TileLoad { buffer, origin, .. } => {
            let idx = record_node(facts, NodeKind::TileLoad, parent);
            facts
                .buffer_refs
                .push((idx, buffer.duplicate_handle(), BufferRefKind::Read));
            for expr in origin {
                walk_expr(expr, idx, facts);
            }
        }
        Node::TileStore { buffer, origin, .. } => {
            let idx = record_node(facts, NodeKind::TileStore, parent);
            facts
                .buffer_refs
                .push((idx, buffer.duplicate_handle(), BufferRefKind::Write));
            for expr in origin {
                walk_expr(expr, idx, facts);
            }
        }
        Node::TileMatmul { .. } => {
            record_node(facts, NodeKind::TileMatmul, parent);
        }
        Node::TileReduce { .. } => {
            record_node(facts, NodeKind::TileReduce, parent);
        }
        Node::TileElementwise { body, .. } => {
            let idx = record_node(facts, NodeKind::TileElementwise, parent);
            for child in body {
                walk_node(child, Some(idx), facts);
            }
        }
        Node::TileDecl { .. } => {
            record_node(facts, NodeKind::TileDecl, parent);
        }
        Node::Opaque(_) => {
            record_node(facts, NodeKind::Opaque, parent);
        }
    }
}

fn walk_expr(expr: &Expr, owning_node: NodeIndex, facts: &mut ProgramFacts) {
    match expr {
        Expr::Var(name) => {
            facts.var_reads.push((owning_node, name.duplicate_handle()));
        }
        Expr::Load { buffer, index } => {
            facts
                .buffer_refs
                .push((owning_node, buffer.duplicate_handle(), BufferRefKind::Read));
            walk_expr(index, owning_node, facts);
        }
        Expr::BufLen { buffer } => {
            facts
                .buffer_refs
                .push((owning_node, buffer.duplicate_handle(), BufferRefKind::Read));
        }
        // A callee parameter is a read-only or uniform buffer by
        // declaration, so passing a buffer to a callee is a read of it.
        Expr::BufferRef { buffer } => {
            facts
                .buffer_refs
                .push((owning_node, buffer.duplicate_handle(), BufferRefKind::Read));
        }
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ..
        } => {
            facts.buffer_refs.push((
                owning_node,
                buffer.duplicate_handle(),
                BufferRefKind::Atomic(*op),
            ));
            walk_expr(index, owning_node, facts);
            if let Some(e) = expected.as_deref() {
                walk_expr(e, owning_node, facts);
            }
            walk_expr(value, owning_node, facts);
        }
        Expr::BinOp { left, right, .. } => {
            walk_expr(left, owning_node, facts);
            walk_expr(right, owning_node, facts);
        }
        Expr::UnOp { operand, .. } => walk_expr(operand, owning_node, facts),
        Expr::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, owning_node, facts);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            walk_expr(cond, owning_node, facts);
            walk_expr(true_val, owning_node, facts);
            walk_expr(false_val, owning_node, facts);
        }
        Expr::Cast { value, .. } | Expr::SubgroupReduce { value, .. } => {
            walk_expr(value, owning_node, facts);
        }
        Expr::Fma { a, b, c } => {
            walk_expr(a, owning_node, facts);
            walk_expr(b, owning_node, facts);
            walk_expr(c, owning_node, facts);
        }
        Expr::SubgroupBallot { cond } => walk_expr(cond, owning_node, facts),
        Expr::SubgroupShuffle { value, lane } => {
            walk_expr(value, owning_node, facts);
            walk_expr(lane, owning_node, facts);
        }
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize
        | Expr::Opaque(_) => {}
    }
}
