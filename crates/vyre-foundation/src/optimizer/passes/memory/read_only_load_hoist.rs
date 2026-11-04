//! buffer aliasing facts into load elision.
//!
//! Read-only-buffer slice shipped here. When both arms of an
//! `Node::If` begin with a `Let(name, Load(buf, idx))` whose
//! `buf` is declared `BufferAccess::ReadOnly` AND the same name +
//! same index, the Load is hoisted before the If. The ReadOnly
//! declaration is the alias proof: a ReadOnly buffer is fully
//! initialised by the host before kernel launch, so the Load is
//! observably-safe to execute on the unconditional path  -  there
//! is no observable difference between "load was already issued"
//! and "load was about to be issued in one arm only".
//!
//! Op id: `vyre-foundation::optimizer::passes::read_only_load_hoist`.
//! Soundness: `Exact`. The ReadOnly access mode is enforced by the
//! buffer table; any pass that mutates a ReadOnly buffer is a
//! validation error caught by `Program::validate()`. Therefore the
//! Load result is invariant under the If's two execution paths,
//! and hoisting the Load to the unconditional path produces the
//! same value at every read site.
//!
//! Cost direction: monotone-down on `node_count` (one fewer Let
//! per fired hoist) and monotone-down on per-arm dispatch overhead
//! (the Load is issued once instead of once per branch).
//!
//! Preserves: every analysis. Invalidates: nothing  -  the hoisted
//! Load is the alias-proof-licensed counterpart of A18's
//! observably-free prefix hoist for non-Load values.
//!
//! ## Pattern
//!
//! ```text
//! If(cond,
//!    [Let(x, Load(ro_buf, idx)), then_rest...],
//!    [Let(x, Load(ro_buf, idx)), other_rest...])
//!     where program.buffer(ro_buf).access() == BufferAccess::ReadOnly
//!     AND idx is observably-free
//! → Let(x, Load(ro_buf, idx)); If(cond, [then_rest...], [other_rest...])
//! ```
//!
//! Idx must be observably-free because the index expression also
//! becomes unconditional after the hoist.
//!
//! ## Why this is A15
//!
//! A15 says "buffer aliasing facts into load elision". The full
//! alias substrate (proving two arbitrary buffers don't alias) is
//! a downstream alias analysis. ReadOnly is the trivial alias proof: a buffer
//! that nobody writes cannot alias with any write target, so its
//! Loads are invariant across control flow. Shipping the trivial
//! slice here gives the hot path the same code-size win that the
//! full aliasing substrate would deliver, while the fact-driven
//! variant lands beside the downstream alias pass.

use crate::ir::{BufferAccess, Expr, Ident, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::passes::expr_is_observably_free;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::visit::bound_names::count_bound_names;
use rustc_hash::{FxHashMap, FxHashSet};

/// Hoist Loads on declared-ReadOnly buffers out of common
/// branch prefixes.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "read_only_load_hoist",
    requires = [],
    invalidates = [],
    phase = "memory",
    boundary_class = "abi_preserving",
    cost_model_family = "memory"
)]
pub struct ReadOnlyLoadHoistPass;

impl ReadOnlyLoadHoistPass {
    /// Skip programs with no candidate `If`.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        // A hoist needs an If whose two arms both open with a load from a
        // ReadOnly buffer. The kind filter is O(1); collecting the ReadOnly
        // buffer names is not, so it comes second.
        let required = [crate::ir::stats::NODE_KIND_IF];
        if !driver::carries_every_kind(program, &required) {
            return PassAnalysis::SKIP;
        }
        let read_only = read_only_buffer_set(program);
        if read_only.is_empty() {
            return PassAnalysis::SKIP;
        }
        driver::analyze_candidates(program, &required, &mut |node| {
            opens_hoistable_pair(node, &read_only)
        })
    }

    /// Walk the entry tree and hoist common Read-Only-Load prefixes.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        let read_only = read_only_buffer_set(&program);
        if read_only.is_empty() {
            return PassResult {
                program,
                changed: false,
            };
        }
        driver::rewrite_entry_bodies(program, &mut |body| hoist_in_body(body, &read_only))
    }
}

fn read_only_buffer_set(program: &Program) -> FxHashSet<Ident> {
    program
        .buffers()
        .iter()
        .filter(|b| matches!(b.access(), BufferAccess::ReadOnly))
        .map(|b| Ident::from(b.name.as_ref()))
        .collect()
}

/// `body` with every hoistable read-only load prefix lifted out of the `If` it
/// opened both arms of, or `None` when no `If` here has one.
///
/// This is a body rule rather than a node rule because scope safety is a
/// property of the enclosing sequence, not of the `If`: see
/// [`hoistable_prefix_len`].
fn hoist_in_body(body: &[Node], read_only: &FxHashSet<Ident>) -> Option<Vec<Node>> {
    // Structural screen before the name census: no arm pair opens with the same
    // read-only load, so nothing here can hoist and the body stays borrowed.
    if !body
        .iter()
        .any(|node| opens_hoistable_pair(node, read_only))
    {
        return None;
    }
    let mut counts: FxHashMap<Ident, usize> = FxHashMap::default();
    count_bound_names(body, &mut counts);
    let prefixes: Vec<usize> = body
        .iter()
        .map(|node| hoistable_prefix_len(node, read_only, &counts))
        .collect();
    if prefixes.iter().all(|len| *len == 0) {
        return None;
    }

    let mut out = Vec::with_capacity(body.len());
    for (node, prefix_len) in body.iter().zip(prefixes) {
        match node {
            Node::If {
                cond,
                then,
                otherwise,
            } if prefix_len > 0 => {
                out.extend(then[..prefix_len].iter().cloned());
                out.push(Node::If {
                    cond: cond.clone(),
                    then: then[prefix_len..].to_vec(),
                    otherwise: otherwise[prefix_len..].to_vec(),
                });
            }
            other => out.push(other.clone()),
        }
    }
    Some(out)
}

/// How many leading nodes `node`'s arms share that this body may lift out:
/// pairs that are structurally hoistable and bound by nothing but those arms.
///
/// Hoisting `let x = load(ro, i)` out of both arms moves `x` from arm scope,
/// which the block-scoped IR pops at arm exit, into the enclosing body, where it
/// then lives across the `If` and every later sibling. That is sound only if no
/// other node in the body binds `x`; otherwise the hoisted binding collides with
/// that binder, which the validator rejects as a duplicate sibling (V032) or a
/// shadow (V008). A name bound at the front of both arms is counted exactly
/// twice over the body iff this `If` is its only binder, so `counts[x] == 2` is
/// the scope-safety gate.
fn hoistable_prefix_len(
    node: &Node,
    read_only: &FxHashSet<Ident>,
    counts: &FxHashMap<Ident, usize>,
) -> usize {
    let Node::If {
        then, otherwise, ..
    } = node
    else {
        return 0;
    };
    driver::common_prefix_len(then, otherwise, |t, o| {
        is_hoistable_load_pair(t, o, read_only)
            && matches!(t, Node::Let { name, .. } if counts.get(name).copied().unwrap_or(0) == 2)
    })
}

/// True iff both arms of `node` open with the same hoistable read-only load.
///
/// The structural half of the test, and the one the analysis uses: scope safety
/// needs the enclosing body, which the analysis walk does not carry, so
/// `analyze_impl` may over-approximate `RUN` into a no-op transform.
fn opens_hoistable_pair(node: &Node, read_only: &FxHashSet<Ident>) -> bool {
    let Node::If {
        then, otherwise, ..
    } = node
    else {
        return false;
    };
    match (then.first(), otherwise.first()) {
        (Some(t), Some(o)) => is_hoistable_load_pair(t, o, read_only),
        _ => false,
    }
}

/// True iff both nodes bind the SAME name to the SAME read-only load with an
/// observably-free index.
fn is_hoistable_load_pair(a: &Node, b: &Node, read_only: &FxHashSet<Ident>) -> bool {
    let Node::Let {
        name: name_a,
        value: value_a,
    } = a
    else {
        return false;
    };
    let Node::Let {
        name: name_b,
        value: value_b,
    } = b
    else {
        return false;
    };
    if name_a != name_b || value_a != value_b {
        return false;
    }
    matches!(value_a, Expr::Load { buffer, index } if read_only.contains(buffer) && expr_is_observably_free(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Expr, Ident, Node};

    fn ro_buf(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 0, BufferAccess::ReadOnly, DataType::U32).with_count(8)
    }

    fn rw_buf(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 1, BufferAccess::ReadWrite, DataType::U32).with_count(8)
    }

    fn program(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Program {
        Program::wrapped(buffers, [1, 1, 1], entry)
    }

    fn find_siblings(nodes: &[Node]) -> Option<&[Node]> {
        if nodes
            .iter()
            .any(|n| matches!(n, Node::Let { .. } | Node::If { .. }))
        {
            return Some(nodes);
        }
        for n in nodes {
            let body = match n {
                Node::Block(body) => body.as_slice(),
                Node::Region { body, .. } => body.as_ref().as_slice(),
                _ => continue,
            };
            if let Some(found) = find_siblings(body) {
                return Some(found);
            }
        }
        None
    }

    /// Positive: Load on a ReadOnly buffer at the start of both arms
    /// hoists out before the If.
    #[test]
    fn hoists_read_only_load_prefix() {
        let load = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(0)),
        };
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![
                Node::let_bind("x", load.clone()),
                Node::store("rw", Expr::u32(0), Expr::var("x")),
            ],
            otherwise: vec![
                Node::let_bind("x", load),
                Node::store("rw", Expr::u32(1), Expr::var("x")),
            ],
        }];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(result.changed, "ReadOnly Load prefix must hoist");
        let siblings =
            find_siblings(result.program.entry()).expect("Fix: hoisted Let + If present");
        assert!(matches!(&siblings[0], Node::Let { name, value }
            if name.as_str() == "x" && matches!(value, Expr::Load { .. })));
        assert!(matches!(&siblings[1], Node::If { .. }));
    }

    /// Negative: Load on a ReadWrite buffer must NOT hoist (alias
    /// proof unavailable; another arm could write between the If and
    /// the post-If sequencing).
    #[test]
    fn keeps_read_write_load() {
        let load = Expr::Load {
            buffer: Ident::from("rw"),
            index: Box::new(Expr::u32(0)),
        };
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![Node::let_bind("x", load.clone())],
            otherwise: vec![Node::let_bind("x", load)],
        }];
        let prog = program(vec![rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(!result.changed, "ReadWrite Load must not hoist");
    }

    /// Negative: differing names block the hoist.
    #[test]
    fn keeps_when_names_differ() {
        let load = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(0)),
        };
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![Node::let_bind("x", load.clone())],
            otherwise: vec![Node::let_bind("y", load)],
        }];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(!result.changed, "differing names must not hoist");
    }

    /// Negative: differing indices block the hoist.
    #[test]
    fn keeps_when_indices_differ() {
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![Node::let_bind(
                "x",
                Expr::Load {
                    buffer: Ident::from("ro"),
                    index: Box::new(Expr::u32(0)),
                },
            )],
            otherwise: vec![Node::let_bind(
                "x",
                Expr::Load {
                    buffer: Ident::from("ro"),
                    index: Box::new(Expr::u32(1)),
                },
            )],
        }];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);

        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(!result.changed, "differing indices must not hoist");
    }

    /// Negative: an index expression that itself contains a Load
    /// blocks the hoist (the index Load could observe state that
    /// the unconditional path shouldn't trigger).
    #[test]
    fn keeps_when_index_reads_memory() {
        let load = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::Load {
                buffer: Ident::from("rw"),
                index: Box::new(Expr::u32(0)),
            }),
        };
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![Node::let_bind("x", load.clone())],
            otherwise: vec![Node::let_bind("x", load)],
        }];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(!result.changed, "index that reads memory must block hoist");
    }

    /// `analyze` short-circuits when the program declares no
    /// ReadOnly buffer.
    #[test]
    fn analyze_skips_program_with_no_read_only_buffer() {
        let entry = vec![Node::store("rw", Expr::u32(0), Expr::u32(1))];
        let prog = program(vec![rw_buf("rw")], entry);
        match crate::optimizer::ProgramPass::analyze(&ReadOnlyLoadHoistPass, &prog) {
            PassAnalysis::SKIP => {}
            other => panic!("expected SKIP, got {other:?}"),
        }
    }

    /// Positive end-to-end smoke: chain of two ReadOnly Loads with
    /// different indices in the prefix hoists both.
    #[test]
    fn hoists_chain_of_read_only_loads() {
        let load_a = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(0)),
        };
        let load_b = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(1)),
        };
        let entry = vec![Node::If {
            cond: Expr::var("c"),
            then: vec![
                Node::let_bind("a", load_a.clone()),
                Node::let_bind("b", load_b.clone()),
                Node::store("rw", Expr::u32(0), Expr::var("a")),
            ],
            otherwise: vec![
                Node::let_bind("a", load_a),
                Node::let_bind("b", load_b),
                Node::store("rw", Expr::u32(1), Expr::var("b")),
            ],
        }];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(result.changed, "chain of ReadOnly Loads must hoist");
        let siblings =
            find_siblings(result.program.entry()).expect("Fix: hoisted Lets + If present");
        assert!(siblings.len() >= 3);
        assert!(matches!(&siblings[0], Node::Let { name, .. } if name.as_str() == "a"));
        assert!(matches!(&siblings[1], Node::Let { name, .. } if name.as_str() == "b"));
    }

    /// Negative (scope extension): the hoisted name `x` is rebound by a later
    /// sibling. Hoisting moves `x` from arm scope to the enclosing scope, where
    /// it lives across the If and collides with the trailing `let x` -- a
    /// duplicate sibling binding the validator rejects (V032). The pass must
    /// decline. (Oracle-differential proof: tests/read_only_load_hoist_scope.rs.)
    #[test]
    fn keeps_when_hoisted_name_rebound_by_later_sibling() {
        let load = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(0)),
        };
        let entry = vec![
            Node::If {
                cond: Expr::var("c"),
                then: vec![
                    Node::let_bind("x", load.clone()),
                    Node::store("rw", Expr::u32(0), Expr::var("x")),
                ],
                otherwise: vec![
                    Node::let_bind("x", load),
                    Node::store("rw", Expr::u32(1), Expr::var("x")),
                ],
            },
            Node::let_bind("x", Expr::u32(7)), // rebinds `x` after the If
            Node::store("rw", Expr::u32(2), Expr::var("x")),
        ];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(
            !result.changed,
            "hoisting `x` would collide with the later `let x`; pass must decline"
        );
    }

    /// Positive (no over-block): a later sibling that binds a DIFFERENT name
    /// must not block the hoist -- the scope-safety gate keys on the hoisted
    /// name only.
    #[test]
    fn hoists_when_later_sibling_binds_a_different_name() {
        let load = Expr::Load {
            buffer: Ident::from("ro"),
            index: Box::new(Expr::u32(0)),
        };
        let entry = vec![
            Node::If {
                cond: Expr::var("c"),
                then: vec![
                    Node::let_bind("x", load.clone()),
                    Node::store("rw", Expr::u32(0), Expr::var("x")),
                ],
                otherwise: vec![
                    Node::let_bind("x", load),
                    Node::store("rw", Expr::u32(1), Expr::var("x")),
                ],
            },
            Node::let_bind("y", Expr::u32(7)), // different name; no collision
            Node::store("rw", Expr::u32(2), Expr::var("y")),
        ];
        let prog = program(vec![ro_buf("ro"), rw_buf("rw")], entry);
        let result = ReadOnlyLoadHoistPass::transform(prog);
        assert!(
            result.changed,
            "a later sibling binding a different name must not block the hoist"
        );
        let siblings = find_siblings(result.program.entry()).expect("hoisted Let + If present");
        assert!(matches!(&siblings[0], Node::Let { name, .. } if name.as_str() == "x"));
    }
}
