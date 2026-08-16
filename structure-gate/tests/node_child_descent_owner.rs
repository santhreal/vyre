//! The class closed here: a hand-written IR walk whose catch-all arm doubles as
//! "this variant has no children".
//!
//! # What this is not
//!
//! It is not a count of `match node {` blocks. That was tried, as a row named
//! `P-DELETE-1` in the retired shell unification ratchet, and it measured the
//! wrong thing in both directions: 22 distinct traversals over one enum is not
//! duplication, `match node {}` is the only idiom Rust offers for dispatching on
//! a variant, and deleting an EXHAUSTIVE block lowered the count while making the
//! workspace less safe. It is also not a count of catch-all arms. Most catch-alls
//! are correct: an optimizer pass that recognises one node kind to decide whether
//! to rewrite it is right to ignore every other kind, and `_ => {}` is the right
//! arm there. A gate that flagged those would flag mostly-correct code and be
//! switched off, which is how `P-DELETE-1` died.
//!
//! # The property
//!
//! The dangerous shape is narrower and it is mechanical. A block is reported when
//! all three hold:
//!
//! 1. it dispatches on a node (`match node {`, or the `&`/`*`/`.as_ref()` forms),
//! 2. it has a top-level `_ =>` arm, and
//! 3. it destructures a child-body field (`then`, `otherwise`, `body`, `nodes`)
//!    out of a `Node::` pattern and recurses on it.
//!
//! Together those mean the block derives child structure itself and then declares,
//! through the catch-all, that every variant it was not told about is a leaf. Add
//! a nesting variant to `Node` and such a block silently stops descending. In
//! `validate::barrier` that is a race rather than a missed optimization: a barrier
//! inside an unrecognised variant makes an exit look ordered when it is not.
//!
//! # Where children come from instead
//!
//! Three owners, one per Rust reference mode, each an exhaustive match with no
//! catch-all so a new variant fails to COMPILE there:
//!
//! - `visit::child_bodies` for a shared read,
//! - `visit::child_bodies_mut` for a move or an in-place mutation,
//! - `transform::rewrite_walk::rewrite_node` for a borrow-preserving rebuild.
//!
//! A block that takes its children from one of those is not reported, however
//! many catch-alls it carries, because the catch-all is then a decision about
//! what to DO and no longer a claim about what a variant contains.
//!
//! # Why it fails by default
//!
//! [`WAIVERS`] is a closed roster: a reported site absent from it fails, and a
//! roster entry that matches nothing fails as stale. So a new hand-written descent
//! fails without anyone remembering this file exists, and a descent that is fixed
//! cannot leave its waiver behind to cover the next one. The roster is the debt
//! ledger, not a suppression list: each entry names the subsystem that owns the
//! file, because the file's owner is the only agent permitted to edit it.
//!
//! # What it does not catch
//!
//! A walk that reaches children through a helper in another file, and a walk over
//! `Expr` rather than `Node`. `Expr` descent is held by the compile-time owners
//! plus `vyre-foundation/tests/ir_variant_shape_owner_closure.rs`, which drives
//! fixtures derived from `EXPR_VARIANT_NAMES` at run time.

use std::collections::BTreeSet;
use std::path::Path;

use structure_gate::source_scan::{
    is_word_byte, mask_comments_and_strings, matching_brace, rust_sources_with_text,
};
use structure_gate::workspace_root;

/// Field names that hold child nodes on some `Node` variant.
///
/// Read off the three owners in `vyre-foundation/src/transform/visit/mod.rs` and
/// `rewrite_walk.rs`. A `Node` variant that gains a body slot under a new field
/// name has to be added here, and the owners fail to compile until somebody
/// looks at them, which is when that happens.
const CHILD_BODY_FIELDS: [&str; 4] = ["then", "otherwise", "body", "nodes"];

/// Call paths that mean "somebody else enumerated the variants for me".
const OWNER_CALLS: [&str; 5] = [
    "child_bodies",
    "child_bodies_mut",
    "rewrite_node",
    "rewrite_body",
    "any_descendant",
];

/// Sites that still derive child structure themselves, with the subsystem that
/// owns the file and why it has not been converted.
///
/// Adding a row is a decision, not a formality: the row says a hand-written
/// descent is acceptable HERE, and the reason has to say what makes it
/// acceptable. "Not converted yet" is a valid reason exactly once, while the
/// conversion is in flight in another lane, and the lane that converts the file
/// deletes the row in the same commit or this gate fails as stale.
const WAIVERS: &[Waiver] = &[
    Waiver {
        path: "conform/vyre-conform/tests/contract_cases/composition_discipline__every_op_is_under_complexity_budget.rs",
        owner: "ToolingFrontend",
        reason: "conform harness walk, converted with the rest of the conform contract cases",
    },
    Waiver {
        path: "conform/vyre-conform/tests/contract_cases/composition_discipline__measure_program.rs",
        owner: "ToolingFrontend",
        reason: "conform harness walk, converted with the rest of the conform contract cases",
    },
    Waiver {
        path: "conform/vyre-conform/tests/contract_cases/parity_matrix__synthetic_entries.rs",
        owner: "ToolingFrontend",
        reason: "conform harness walk, converted with the rest of the conform contract cases",
    },
    Waiver {
        path: "vyre-driver-wgpu/tests/op_pairwise/all_entries_vec.rs",
        owner: "Backends",
        reason: "driver test walk, converted with the driver walker lane",
    },
    Waiver {
        path: "vyre-libs/src/nn/linear/inner/linear_4bit/affine_grouped.rs",
        owner: "Backends",
        reason: "domain op builder walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/src/security/aliases_dataflow.rs",
        owner: "Backends",
        reason: "domain dataflow walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/blake3_kat.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/indexed_map_composition_contracts.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/loop_unroll_trip1_idempotence.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/math_algebra_branchless_contracts.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/nn_attention_clone_family_ir_invariance.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/optimized_programs.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/parsing_walker_clone_family.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/region_chain_invariant.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-libs/tests/workgroup_cooperative_tiling.rs",
        owner: "Backends",
        reason: "domain test walk, owned by the vyre-libs lane",
    },
    Waiver {
        path: "vyre-reference/src/execution/hashmap/step/node_step.rs",
        owner: "Backends",
        reason: "the reference evaluator interprets each variant, so its dispatch is the decision, not a descent shortcut",
    },
    Waiver {
        path: "vyre-reference/src/execution/node.rs",
        owner: "Backends",
        reason: "the reference evaluator interprets each variant, so its dispatch is the decision, not a descent shortcut",
    },
    Waiver {
        path: "vyre-runtime/src/resident_work_queue/planner/barriers.rs",
        owner: "Backends",
        reason: "barrier planning; a missed nested barrier is a race, highest-priority conversion",
    },
    Waiver {
        path: "xtask-registry/src/docs/operation_schema.rs",
        owner: "ToolingFrontend",
        reason: "tooling walk over IR for documentation generation",
    },
    Waiver {
        path: "xtask-registry/src/gates/lego_audit/fingerprint.rs",
        owner: "ToolingFrontend",
        reason: "tooling walk over IR for a gate, emitting one fingerprint byte per node kind, so the arm set is the measurement",
    },
    Waiver {
        path: "xtask-registry/src/gates/lego_audit/ops.rs",
        owner: "ToolingFrontend",
        reason: "tooling walk over IR for a gate, counting nodes per variant, so the arm set is the measurement",
    },
    Waiver {
        path: "xtask-registry/src/print_composition.rs",
        owner: "ToolingFrontend",
        reason: "tooling walk over IR for a report",
    },
    Waiver {
        path: "vyre-debug/src/source_assignments.rs",
        owner: "CompilerCore",
        reason: "lexical scope stack pushed per child body; the owner walk carries no scope frame",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/algebraic/const_fold/binop_identities.rs",
        owner: "CompilerCore",
        reason: "rebuild threading a local literal environment through each child in order",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/cleanup/branch_value_hoist.rs",
        owner: "CompilerCore",
        reason: "test helper searching for one structural shape, deliberately independent of the walker under test",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/cleanup/empty_block_collapse.rs",
        owner: "CompilerCore",
        reason: "test oracle counting empty blocks independently of the pass it checks",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/cleanup/region_fusion_hint.rs",
        owner: "CompilerCore",
        reason: "reads sibling windows inside each body, which is position information the owner walk drops",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/fusion_cse/fusion/mod.rs",
        owner: "CompilerCore",
        reason: "rebuild constructing each variant with its own builder; belongs on rewrite_node, conversion in flight",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/loops/loop_redundant_bound_check_elide.rs",
        owner: "CompilerCore",
        reason: "two remaining blocks are test oracles counting guards and stores independently of the pass",
    },
    Waiver {
        path: "vyre-foundation/src/optimizer/passes/loops/loop_unroll.rs",
        owner: "CompilerCore",
        reason: "catch-all encodes the scope claim that other variants open their own scope, documented at the site",
    },
    Waiver {
        path: "vyre-foundation/src/validate/rule_pipeline/mod.rs",
        owner: "CompilerCore",
        reason: "explicit work stack assigning per child body a divergence flag and a depth the owner walk has no slot for",
    },
    Waiver {
        path: "vyre-foundation/tests/adversarial_loop_induction_rebind.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/canonical_determinism.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/collective_ir_contracts.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/contract_cases/autodiff_transform_contracts_support.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/contract_cases/program_stats_proptest__arb_node.rs",
        owner: "CompilerCore",
        reason: "generator building arbitrary nodes; routing it through the owner would compare the walker against itself",
    },
    Waiver {
        path: "vyre-foundation/tests/fusion_substitute_into_subgroup_operand.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the substitution walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/inline_buffer_reference_arguments.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the inliner walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/inline_callee_local_rename_in_trap_and_async.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the inliner walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/rewrite_driver_descends_into_async_offset.rs",
        owner: "CompilerCore",
        reason: "test oracle whose whole point is checking the rewrite driver against a separate descent",
    },
    Waiver {
        path: "vyre-foundation/tests/subst_preserves_subgroup_reduce_op.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the substitution walker it audits",
    },
    Waiver {
        path: "vyre-foundation/tests/wire_buffer_ref_round_trip.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the wire round trip it audits",
    },
    Waiver {
        path: "vyre-libs/tests/surface_contracts.rs",
        owner: "Backends",
        reason: "library surface contract walk owned by the libs lane, not converted in this lane",
    },
    Waiver {
        path: "vyre-pass-engine/tests/dce_program_back_edge_contract.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the dead code walker it audits",
    },
    Waiver {
        path: "vyre-primitives/src/graph/dominator_tree/tests/mod.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the dominator construction it audits",
    },
    Waiver {
        path: "vyre-primitives/src/graph/persistent_bfs/tests/behavior_contracts/program_sync_contracts.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the traversal it audits",
    },
    Waiver {
        path: "vyre-primitives/src/graph/persistent_bfs/tests/validation_and_builders.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the traversal it audits",
    },
    Waiver {
        path: "vyre-primitives/tests/adversarial_math.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-primitives/tests/ir_shape/mod.rs",
        owner: "CompilerCore",
        reason: "shape oracle for primitive graph tests, independent of the production walker by design",
    },
    Waiver {
        path: "vyre-primitives/tests/loop_back_edge_audit.rs",
        owner: "CompilerCore",
        reason: "back edge audit oracle, independent of the production walker by design",
    },
];

/// One recorded hand-written descent.
struct Waiver {
    path: &'static str,
    owner: &'static str,
    reason: &'static str,
}

/// A reported block: the file it sits in and the line its `match` opens on.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Site {
    path: String,
    line: usize,
}

#[test]
fn no_unrecorded_walk_derives_node_children_itself() {
    let root = workspace_root();
    let sites = scan(&root);

    let waived: BTreeSet<&str> = WAIVERS.iter().map(|waiver| waiver.path).collect();
    assert_eq!(
        waived.len(),
        WAIVERS.len(),
        "Fix: the waiver roster lists a path twice, so one of the two reasons is unread"
    );

    let unrecorded: Vec<&Site> = sites
        .iter()
        .filter(|site| !waived.contains(site.path.as_str()))
        .collect();
    assert!(
        unrecorded.is_empty(),
        "{} walk(s) derive `Node` child structure themselves and end in a catch-all arm, so a \
         nesting variant added to `Node` would be skipped there in silence:\n{}\n\n\
         Fix: take the children from one owner instead. \
         `vyre_foundation::visit::child_bodies` for a shared read, \
         `child_bodies_mut` for a move or in-place mutation, \
         `vyre_foundation::transform::rewrite_walk::rewrite_node` for a borrow-preserving \
         rebuild. A predicate becomes \
         `<per-variant decision> || child_bodies(node).into_iter().flatten().any(recurse)`. \
         Keep the arms that make a real per-variant decision; delete only the arms whose whole \
         content is descending into that variant's children. If the block is a deliberate \
         independent oracle, add it to WAIVERS in {} with the reason that makes it one.",
        unrecorded.len(),
        unrecorded
            .iter()
            .map(|site| format!("  {}:{}", site.path, site.line))
            .collect::<Vec<_>>()
            .join("\n"),
        file!(),
    );
}

/// A waiver whose file no longer contains the shape it waives is deleted.
///
/// Without this half the roster is a suppression list: a converted file keeps
/// its row, and the row then covers the NEXT hand-written descent added to that
/// file, which is the failure mode of every allowlist that only grows.
#[test]
fn every_waiver_still_describes_a_real_walk() {
    let root = workspace_root();
    let reported: BTreeSet<String> = scan(&root).into_iter().map(|site| site.path).collect();

    let stale: Vec<&Waiver> = WAIVERS
        .iter()
        .filter(|waiver| !reported.contains(waiver.path))
        .collect();
    assert!(
        stale.is_empty(),
        "{} waiver(s) no longer match anything. Either the file was converted to the shared \
         owner, in which case delete the row, or the file was moved or deleted, in which case \
         the row is measuring nothing:\n{}",
        stale.len(),
        stale
            .iter()
            .map(|waiver| format!(
                "  {} (owner {}): {}",
                waiver.path, waiver.owner, waiver.reason
            ))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// Every waiver names an owning subsystem and a reason that says something.
#[test]
fn every_waiver_names_an_owner_and_a_reason() {
    for waiver in WAIVERS {
        assert!(
            matches!(
                waiver.owner,
                "CompilerCore" | "Backends" | "ToolingFrontend"
            ),
            "Fix: waiver for {} names owner `{}`, which is not a subsystem. Only the subsystem \
             that owns the file may convert it, so the row has to say which one.",
            waiver.path,
            waiver.owner
        );
        assert!(
            waiver.reason.split_whitespace().count() >= 5,
            "Fix: waiver for {} has no reason worth reading: {:?}",
            waiver.path,
            waiver.reason
        );
    }
}

/// The scanner sees the shape it claims to see, and does not see the shapes it
/// claims to allow.
///
/// Held against literal sources rather than against the tree, so a tree that
/// happens to be clean cannot make this pass by measuring nothing.
#[test]
fn the_scanner_reports_the_defect_and_not_its_correct_neighbours() {
    let defect = "\
fn walk(node: &Node) -> bool {
    match node {
        Node::Barrier { .. } => true,
        Node::If { then, otherwise, .. } => {
            then.iter().any(walk) || otherwise.iter().any(walk)
        }
        Node::Loop { body, .. } => body.iter().any(walk),
        _ => false,
    }
}
";
    assert_eq!(
        blocks_in(defect).len(),
        1,
        "Fix: the scanner stopped seeing a hand-written descent with a catch-all"
    );

    let routed = "\
fn walk(node: &Node) -> bool {
    let here = match node {
        Node::Barrier { .. } => true,
        _ => false,
    };
    here || child_bodies(node).into_iter().flatten().any(walk)
}
";
    assert!(
        blocks_in(routed).is_empty(),
        "Fix: the scanner reports a walk that already takes its children from the owner, which \
         is the shape it exists to encourage"
    );

    let exhaustive = "\
fn walk(node: &Node) -> bool {
    match node {
        Node::If { then, otherwise, .. } => then.iter().any(walk) || otherwise.iter().any(walk),
        Node::Loop { body, .. } => body.iter().any(walk),
        Node::Return => false,
    }
}
";
    assert!(
        blocks_in(exhaustive).is_empty(),
        "Fix: the scanner reports an exhaustive descent. Exhaustive is the safe shape: adding a \
         variant fails to compile there, which is the outcome this gate wants"
    );

    let decision_only = "\
fn rewrite(node: &Node) -> Option<Node> {
    match node {
        Node::Loop { body, .. } if body.is_empty() => None,
        _ => Some(node.clone()),
    }
}
";
    assert!(
        blocks_in(decision_only).is_empty(),
        "Fix: the scanner reports a pass that reads a child body to make a decision without \
         recursing. That is not a descent and flagging it is how the previous ratchet died"
    );
}

/// Every reported block in the tree, ordered.
fn scan(root: &Path) -> Vec<Site> {
    let mut sites = Vec::new();
    for (relative, text) in rust_sources_with_text(root) {
        for line in blocks_in(&text) {
            sites.push(Site {
                path: relative.clone(),
                line,
            });
        }
    }
    sites.sort();
    sites
}

/// Opening line of every reported block in `text`.
fn blocks_in(text: &str) -> Vec<usize> {
    let source = mask_comments_and_strings(text);
    let bytes = source.as_bytes();
    let mut reported = Vec::new();
    let mut cursor = 0;

    while let Some(offset) = source[cursor..].find("match ") {
        let start = cursor + offset;
        cursor = start + "match ".len();
        let Some(open) = scrutinee_is_a_node(&source[cursor..]) else {
            continue;
        };
        let brace = cursor + open;
        let Some(end) = matching_brace(bytes, brace) else {
            continue;
        };
        let body = &source[brace + 1..end];
        if has_top_level_wildcard_arm(body)
            && destructures_a_child_body(body)
            && recurses_on_a_child_body(body)
            && !OWNER_CALLS.iter().any(|call| body.contains(call))
        {
            reported.push(source[..start].matches('\n').count() + 1);
        }
    }
    reported
}

/// Offset of the opening brace when the scrutinee is a node, else `None`.
///
/// The `&`, `*` and `.as_ref()` forms are the same dispatch: a walk that writes
/// `match &node {` is not exempt from the property because of an ampersand.
fn scrutinee_is_a_node(rest: &str) -> Option<usize> {
    let mut consumed = 0;
    let mut tail = rest;

    let trimmed = tail.trim_start_matches([' ', '&', '*']);
    consumed += tail.len() - trimmed.len();
    tail = trimmed.strip_prefix("node")?;
    consumed += "node".len();

    if let Some(after) = tail.strip_prefix(".as_ref()") {
        consumed += ".as_ref()".len();
        tail = after;
    }

    let head = tail.trim_start();
    if !head.starts_with('{') {
        return None;
    }
    Some(consumed + (tail.len() - head.len()))
}

/// True when `body` has a `_ =>` arm at arm depth, not inside a nested pattern.
fn has_top_level_wildcard_arm(body: &str) -> bool {
    let bytes = body.as_bytes();
    let mut depth = 0i32;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => depth -= 1,
            b'_' if depth == 0 => {
                let before_is_word = index
                    .checked_sub(1)
                    .is_some_and(|previous| is_word_byte(bytes[previous]));
                if before_is_word {
                    continue;
                }
                let after = body[index + 1..].trim_start();
                let after = match after.strip_prefix("if ") {
                    Some(guard) => guard.split_once("=>").map_or("", |(_, tail)| tail),
                    None => after,
                };
                if after.starts_with("=>") || after.is_empty() {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// True when some `Node::` pattern in `body` binds a child-body field.
fn destructures_a_child_body(body: &str) -> bool {
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find("Node::") {
        let start = cursor + offset;
        cursor = start + "Node::".len();
        let Some(open) = body[cursor..].find('{') else {
            break;
        };
        let Some(close) = body[cursor + open..].find('}') else {
            break;
        };
        // Only a pattern, not a construction: a construction has an `=>` or a
        // `return` between the variant name and its brace on the same arm.
        let fields = &body[cursor + open + 1..cursor + open + close];
        if fields.split(',').any(|field| {
            let name = field.split(':').next().unwrap_or("").trim();
            CHILD_BODY_FIELDS.contains(&name)
        }) {
            return true;
        }
    }
    false
}

/// True when a child-body binder in `body` is iterated or handed to a call.
fn recurses_on_a_child_body(body: &str) -> bool {
    CHILD_BODY_FIELDS.iter().any(|field| {
        let iterated = [
            format!("{field}.iter()"),
            format!("{field}.iter_mut()"),
            format!("in {field}"),
            format!("in &{field}"),
            format!("in {field}.iter()"),
            format!("({field})"),
            format!("(&{field})"),
            format!("({field},"),
            format!("(&{field},"),
        ];
        iterated.iter().any(|form| body.contains(form.as_str()))
    })
}
