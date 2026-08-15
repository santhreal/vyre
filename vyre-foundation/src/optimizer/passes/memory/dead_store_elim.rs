//! `dead_store_elim`  -  drop `Node::Store` whose value is overwritten
//! by a subsequent sibling `Node::Store` to the same `(buffer, index)`
//! before any intervening side-effect could observe the first write.
//!
//! Op id: `vyre-foundation::optimizer::passes::dead_store_elim`.
//! Soundness: `Exact`. `memory::alias` owns the proof that nothing in the
//! gap between the two sibling stores can observe the first write; this pass
//! contributes only the pairing rule and the deletion. Cost direction:
//! monotone-down on `node_count`.
//! Preserves: every analysis. Invalidates: nothing.
//!
//! ## Rule
//!
//! ```text
//! Node::Store { buffer: b, index: i, value: _ }
//! ... straight-line siblings, no nested control flow, no
//!     Load/Atomic/Store/Async*/Indirect/Trap/Barrier touching b ...
//! Node::Store { buffer: b, index: i, value: _ }
//! ```
//!
//! When the matcher fires, the FIRST `Store` is dropped. The second
//! survives and contributes the observable write. Indices are matched
//! by structural equality (literal-aware via `expr_eq`), so dynamic-
//! index stores are kept conservatively. The pass walks recursively
//! through `If`/`Loop`/`Block`/`Region` containers but only fires on
//! sibling sequences inside one container; cross-container DSE is left
//! to a stronger reaching-store analysis (ROADMAP A22 store-to-load
//! forwarding will produce the alias proof needed for that).
//!
//! Catches:
//!   - generated `Store(buf, 0, x); Store(buf, 0, y);` patterns from
//!     unfused arms or const-fold residue;
//!   - duplicate clears that the host would emit twice if a previous
//!     pass left a redundant initialisation in place.
//!
//! Does not catch (deliberately):
//!   - stores separated by an `If` whose branches don't touch the
//!     buffer (would need branch-aware reaching-store analysis);
//!   - stores to overlapping but not equal indices (no alias model
//!     yet);
//!   - stores where the value of the first one is later read via
//!     `Load(buffer, *)`  -  `alias::Interference::Reads` keeps the first
//!     store alive when any node between the two reads from `buffer`,
//!     INCLUDING the overwriting store's own index/value subexpressions
//!     (a read-modify-write like `Store(b,0, Load(b,0)+5)` reads the
//!     first store before overwriting it, so the first store is live).

use super::alias::{expr_touches_buffer, node_interferes, Interference};
use crate::ir::{Ident, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};

/// Drop straight-line `Node::Store` values that are overwritten by the
/// next sibling `Node::Store` to the same `(buffer, index)` with no
/// observable read in between.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "dead_store_elim",
    requires = [],
    invalidates = [],
    phase = "memory",
    boundary_class = "abi_preserving",
    cost_model_family = "memory"
)]
pub struct DeadStoreElim;

impl DeadStoreElim {
    /// Skip the pass when no body in the program contains two stores
    /// to the same buffer that *could* alias each other.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        driver::analyze_candidate_bodies(
            program,
            &[crate::ir::stats::NODE_KIND_STORE],
            &mut contains_buffer_pair,
        )
    }

    /// Walk the program tree; remove dead sibling stores in every
    /// sequence body that has them.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        driver::rewrite_entry_bodies(program, &mut drop_dead_stores)
    }
}

/// Remove every `Store(b, i, _)` that has a later sibling `Store(b, i, _)`
/// with no intervening reader of `b` between them.
fn drop_dead_stores(body: &[Node]) -> Option<Vec<Node>> {
    let mut keep = vec![true; body.len()];
    let mut dropped_any = false;
    for first_idx in 0..body.len() {
        if !keep[first_idx] {
            continue;
        }
        let Node::Store {
            buffer: first_buf,
            index: first_idx_expr,
            ..
        } = &body[first_idx]
        else {
            continue;
        };
        // Scan forward over siblings. The `node_interferes` arm below breaks
        // the instant a sibling could observe `first_buf`'s pre-store value, so
        // reaching iteration `second_idx` already proves every earlier sibling
        // took the transparent `_` arm (it did NOT observe `first_buf`).
        // Re-scanning the `[first_idx+1, second_idx)` gap each step would
        // therefore be a provably-redundant O(n^2)-per-store re-walk over the
        // same predicate -- and it re-descends nested `If`/`Loop`/`Region`
        // bodies every step. The gap check lives entirely in the per-sibling
        // match instead.
        for second_idx in (first_idx + 1)..body.len() {
            match &body[second_idx] {
                Node::Store {
                    buffer: second_buf,
                    index: second_idx_expr,
                    value: second_value,
                } if second_buf == first_buf
                    // Structural equality is conservative in the safe
                    // direction: equal indices are equal at run time, and two
                    // that only coincide at run time keep both stores.
                    && first_idx_expr == second_idx_expr
                    // The overwriting store evaluates its index and value
                    // BEFORE writing, so if either reads `first_buf` it observes
                    // the first store's value (e.g. `Store(b,0, Load(b,0)+5)` is
                    // a read-modify-write). Only a write that does NOT read the
                    // buffer makes the earlier store dead. (Without this the
                    // first arm matched on `(buffer,index)` alone and dropped a
                    // store its overwriter still read, a dead-store miscompile.)
                    && !expr_touches_buffer(second_idx_expr, first_buf)
                    && !expr_touches_buffer(second_value, first_buf) =>
                {
                    keep[first_idx] = false;
                    dropped_any = true;
                    break;
                }
                node if node_interferes(node, first_buf, Interference::Reads) => {
                    break;
                }
                _ => {
                    // Keep scanning forward; this sibling is not a
                    // store to (first_buf, first_idx_expr) and does
                    // not touch first_buf either.
                }
            }
        }
    }
    dropped_any.then(|| {
        body.iter()
            .zip(keep)
            .filter_map(|(node, alive)| alive.then(|| node.clone()))
            .collect()
    })
}

/// Whether the program has any sibling pair of stores to the same
/// buffer  -  cheap analysis used by the pass scheduler to skip programs
/// where DSE has nothing to do.
///
/// The candidate is a relation between ADJACENT siblings, so it cannot flatten
/// to a per-node scan: two stores to one buffer are invisible from either store
/// alone. [`driver::analyze_candidate_bodies`] hands the pairwise test each
/// enclosing body, and its slot list comes from `child_bodies`. The
/// hand-written slot match this replaces ended in `_ => return false`, so a
/// redundant pair inside a fifth body-bearing variant read as absent and the
/// pass reported SKIP.
fn contains_buffer_pair(body: &[Node]) -> bool {
    let mut seen: rustc_hash::FxHashSet<&Ident> = rustc_hash::FxHashSet::default();
    for n in body {
        if let Node::Store { buffer, .. } = n {
            if !seen.insert(buffer) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Ident, Node};

    fn buf(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
    }

    fn program_with_entry(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf("buf"), buf("other")], [1, 1, 1], entry)
    }

    fn count_stores(node: &Node) -> usize {
        crate::test_ir_inspect::count_nodes(std::slice::from_ref(node), |candidate| {
            matches!(candidate, Node::Store { .. })
        })
    }

    #[test]
    fn drops_first_of_two_back_to_back_stores_to_same_index() {
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 1, "first dead store must be dropped");
    }

    #[test]
    fn keeps_both_stores_to_different_indices() {
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::u32(1), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(!result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn keeps_both_stores_to_different_buffers() {
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("other", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(!result.changed);
    }

    #[test]
    fn keeps_first_store_when_intervening_load_observes_it() {
        // Store(buf, 0, 1); Let(x, Load(buf, 0)); Store(buf, 0, 2)
        // The Load reads the first store; cannot drop it.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind("x", Expr::load("buf", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "must not drop a store whose value is observably read before the overwrite"
        );
    }

    #[test]
    fn keeps_first_store_when_a_shuffle_lane_loads_it() {
        // Store(buf, 0, 1); Let(x, shuffle(5, Load(buf, 0))); Store(buf, 0, 2)
        // The first store is observed by a Load hidden in the shuffle's LANE
        // operand. `expr_touches_buffer` must descend into the lane; otherwise
        // the read is invisible and the live first store is wrongly eliminated.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind(
                "x",
                Expr::subgroup_shuffle(Expr::u32(5), Expr::load("buf", Expr::u32(0))),
            ),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "a load in a shuffle lane observes the first store; it must not be dropped"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(
            total, 2,
            "both stores must survive when a shuffle lane reads the first"
        );
    }

    #[test]
    fn drops_first_when_intervening_let_reads_a_different_buffer() {
        // The reader touches `other`, not `buf`. Safe to drop the first
        // store to buf.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind("x", Expr::load("other", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 1, "only the overwritten store is removed");
    }

    #[test]
    fn keeps_first_store_when_later_same_buffer_store_index_reads_it() {
        // Store(buf,0,1); Store(buf, Load(buf,0), 2); Store(buf,0,3)
        // The middle store writes to the SAME buffer, but its INDEX reads
        // buf[0] (the first store's value (to compute where it writes)).
        // Dropping the first store would change that index: a miscompile.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::load("buf", Expr::u32(0)), Expr::u32(2)),
            Node::store("buf", Expr::u32(0), Expr::u32(3)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "a same-buffer store whose index reads buf observes the first store"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 3, "all three stores must survive");
    }

    #[test]
    fn keeps_first_store_when_later_same_buffer_store_value_reads_it() {
        // Store(buf,0,1); Store(buf, 9, Load(buf,0)); Store(buf,0,3)
        // The middle store targets a different index of the SAME buffer, but
        // its VALUE reads buf[0] (the first store's value). Dropping the first
        // store would change that value: a miscompile.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::u32(9), Expr::load("buf", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), Expr::u32(3)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "a same-buffer store whose value reads buf observes the first store"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 3, "all three stores must survive");
    }

    #[test]
    fn keeps_first_store_when_barrier_separates_it() {
        // Barriers are observation points (other invocations may read
        // the buffer post-barrier). Conservative: keep the first store.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::barrier(),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "Barrier between stores must keep the first one alive"
        );
    }

    #[test]
    fn keeps_first_store_when_atomic_read_intervenes() {
        // Atomic reads on the same buffer count as observations.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind(
                "x",
                Expr::Atomic {
                    op: AtomicOp::Exchange,
                    buffer: Ident::from("buf"),
                    index: Box::new(Expr::u32(0)),
                    expected: None,
                    value: Box::new(Expr::u32(0)),
                    ordering: crate::ir::MemoryOrdering::Relaxed,
                },
            ),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(!result.changed);
    }

    #[test]
    fn drops_dead_store_inside_if_branch() {
        let entry = vec![Node::if_then(
            Expr::var("c"),
            vec![
                Node::store("buf", Expr::u32(0), Expr::u32(1)),
                Node::store("buf", Expr::u32(0), Expr::u32(2)),
            ],
        )];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn keeps_stores_separated_by_nested_if() {
        // The intervening `If` could read from `buf` in either branch
        //  -  we conservatively keep the first store.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::if_then(Expr::var("c"), vec![Node::Return]),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "nested If between stores is opaque under conservative DSE  -  keep the first"
        );
    }

    #[test]
    fn drops_chain_of_three_redundant_stores() {
        // s1, s2, s3 to (buf, 0): only s3 survives.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
            Node::store("buf", Expr::u32(0), Expr::u32(3)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 1, "only the last store survives");
    }

    #[test]
    fn analyze_skips_program_with_no_redundant_pair() {
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("other", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&DeadStoreElim, &program),
            PassAnalysis::SKIP
        );
    }

    #[test]
    fn analyze_runs_when_redundant_pair_present() {
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&DeadStoreElim, &program),
            PassAnalysis::RUN
        );
    }

    #[test]
    fn keeps_first_store_when_overwriter_value_reads_it() {
        // Store(buf,0,1); Store(buf,0, Load(buf,0)+5). The OVERWRITING store
        // (same buffer+index) reads buf[0] in its value to compute what to
        // write -- a read-modify-write. Dropping the first store changes that
        // read. (Oracle-differential proof:
        // tests/dead_store_elim_overwriter_reads.rs.)
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::store(
                "buf",
                Expr::u32(0),
                Expr::add(Expr::load("buf", Expr::u32(0)), Expr::u32(5)),
            ),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "an overwriter that reads the buffer in its value observes the first store"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 2, "both stores must survive");
    }

    #[test]
    fn keeps_first_store_when_overwriter_index_reads_it() {
        // Store(buf, Load(buf,0), 1); Store(buf, Load(buf,0), 2). The indices
        // are structurally equal, but each reads buf[0] -- which the first
        // store may change -- so the two writes can target different slots.
        // Conservative: keep the first store.
        let entry = vec![
            Node::store("buf", Expr::load("buf", Expr::u32(0)), Expr::u32(1)),
            Node::store("buf", Expr::load("buf", Expr::u32(0)), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "an overwriter whose index reads the buffer observes the first store"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 2, "both stores must survive");
    }

    #[test]
    fn drops_first_across_two_transparent_gap_siblings() {
        // Two siblings that do NOT touch `buf` sit between the dead store and
        // its overwriter. The forward scan treats both as transparent (`_`
        // arm) and still drops the first store -- exercising a multi-sibling
        // gap that the per-sibling `node_observes_buffer` break handles
        // incrementally, with no whole-gap re-scan.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind("x", Expr::load("other", Expr::u32(0))),
            Node::let_bind("y", Expr::u32(7)),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(
            total, 1,
            "first store dies across two buf-transparent siblings"
        );
    }

    #[test]
    fn keeps_first_when_a_non_adjacent_gap_sibling_observes_buf() {
        // The observing reader is NOT the sibling immediately after the first
        // store -- a transparent Let precedes it. The per-sibling break must
        // still fire on the second, observing Let and keep the first store
        // alive. This is exactly the coverage the removed whole-gap re-scan
        // used to provide; the incremental break must subsume it, so this
        // test fails if a future change reintroduces a gap that only checks
        // the immediately-adjacent sibling.
        let entry = vec![
            Node::store("buf", Expr::u32(0), Expr::u32(1)),
            Node::let_bind("x", Expr::load("other", Expr::u32(0))),
            Node::let_bind("y", Expr::load("buf", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), Expr::u32(2)),
        ];
        let program = program_with_entry(entry);
        let result = DeadStoreElim::transform(program);
        assert!(
            !result.changed,
            "an observing reader anywhere in the gap keeps the first store"
        );
        let total: usize = result.program.entry().iter().map(count_stores).sum();
        assert_eq!(total, 2, "both stores survive");
    }
}
