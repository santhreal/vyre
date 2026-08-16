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
//! 3. it binds a child body out of a `Node::` pattern and recurses on it.
//!
//! The child slots are read from the `Node` enum at run time, both the named
//! fields whose type holds nodes and the variants that carry their body in a
//! tuple position, so a variant added under a new field name is scanned for on
//! the same run that declares it.
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
use std::fs;
use std::path::Path;

use structure_gate::source_scan::{
    is_word_byte, mask_comments_and_strings, matching_brace, rust_sources_with_text,
};
use structure_gate::workspace_root;

/// Where `Node` is declared, relative to the workspace root.
///
/// The child slot vocabulary is read out of this file at run time rather than
/// written down here. A list of field names in a test file goes stale the moment
/// a variant arrives under a new name, and a scanner with a stale vocabulary
/// reports nothing while looking like it passed, which is the failure this whole
/// file exists to prevent.
const NODE_ENUM_SOURCE: &str = "vyre-foundation/src/ir_inner/model/generated.rs";

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
        path: "vyre-libs/src/nn/linear/layer/linear_4bit/affine_grouped.rs",
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
        path: "xtask-registry/src/docs/operation_schema/composition.rs",
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
        path: "vyre-foundation/tests/contract_cases/autodiff_transform_contracts_programs.rs",
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
        path: "vyre-libs/src/graph/dominator_tree/tests/mod.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the dominator construction it audits",
    },
    Waiver {
        path: "vyre-libs/src/graph/persistent_bfs/tests/behavior_contracts/program_sync_contracts.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the traversal it audits",
    },
    Waiver {
        path: "vyre-libs/src/graph/persistent_bfs/tests/validation_and_builders.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the traversal it audits",
    },
    Waiver {
        path: "vyre-libs/tests/adversarial_math.rs",
        owner: "CompilerCore",
        reason: "test oracle deliberately independent of the production walker it audits",
    },
    Waiver {
        path: "vyre-libs/tests/ir_shape/mod.rs",
        owner: "CompilerCore",
        reason: "shape oracle for the graph tests, independent of the production walker by design",
    },
    Waiver {
        path: "vyre-libs/tests/loop_back_edge_audit.rs",
        owner: "CompilerCore",
        reason: "back edge audit oracle, independent of the production walker by design",
    },
    Waiver {
        path: "vyre-libs/tests/prefix_scan_contract.rs",
        owner: "CompilerCore",
        reason: "per-lane store counter that folds each condition to decide whether a branch runs for that lane, so it must not descend into a body the lane never takes",
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
    let slots = declared_child_slots(&workspace_root());
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
        blocks_in(defect, &slots).len(),
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
        blocks_in(routed, &slots).is_empty(),
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
        blocks_in(exhaustive, &slots).is_empty(),
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
        blocks_in(decision_only, &slots).is_empty(),
        "Fix: the scanner reports a pass that reads a child body to make a decision without \
         recursing. That is not a descent and flagging it is how the previous ratchet died"
    );
}

/// The vocabulary comes from the enum, and it comes back complete.
///
/// Read against the tree because that is the vocabulary the scan above runs
/// with. The assertion is that every child slot the enum declares is a slot the
/// owner walk destructures, which is the same population from two independent
/// readings: the parser reads the declaration, `child_bodies` states the
/// decision. A slot in the enum that the owner does not name is a body no
/// traversal reaches, and a name in the parser's output that is not a slot means
/// the parser is reading types it should not.
#[test]
fn the_child_slot_vocabulary_is_read_off_the_enum_the_owner_walks() {
    let root = workspace_root();
    let slots = declared_child_slots(&root);
    let owner = root.join("vyre-foundation/src/visit/node_parts.rs");
    let text = fs::read_to_string(&owner).expect("the owner of child bodies is readable");
    let source = mask_comments_and_strings(&text);
    let open = source
        .find("fn child_bodies(")
        .and_then(|at| source[at..].find('{').map(|brace| at + brace))
        .expect("child_bodies has a body");
    let end = matching_brace(source.as_bytes(), open).expect("child_bodies closes");
    let body = &source[open..end];

    let bound = child_body_binders(body, &slots);
    let unnamed: Vec<&String> = slots
        .fields
        .iter()
        .filter(|field| !bound.contains(*field))
        .collect();
    assert!(
        unnamed.is_empty(),
        "Fix: `Node` declares child bodies {unnamed:?} that `child_bodies` does not hand back, so \
         no traversal in the workspace reaches them and the scan cannot tell a walk that misses \
         them from one that does not. Add the slot to the owner in \
         vyre-foundation/src/visit/node_parts.rs."
    );
    let missing_tuple: Vec<&String> = slots
        .tuple_variants
        .iter()
        .filter(|variant| !body.contains(&format!("Node::{variant}(")))
        .collect();
    assert!(
        missing_tuple.is_empty(),
        "Fix: `Node` carries a tuple body on {missing_tuple:?} that `child_bodies` does not \
         destructure"
    );
}

/// A variant that arrives with a child body under a new name is scanned for.
///
/// WHY: the vocabulary used to be four names written in this file. Under that
/// version a variant carrying its body as `arms` was invisible: a hand-written
/// descent over `arms` bound a field the scanner did not know, the block was not
/// reported, and the roster below stayed green while the descent it exists to
/// find was in the tree. The declaration and a descent over it are both injected
/// here, so the proof does not depend on anyone adding a variant.
#[test]
fn a_variant_declaring_a_child_body_under_a_new_name_is_scanned_for() {
    let declaration = "\
vyre_macros::vyre_ast_registry! {
    Node {
        Return,
        Loop { var: Ident, body: Vec<Node> },
        Block(Vec<Node>),
        Speculate { guard: Expr, arms: Vec<Node> },
        Opaque(Arc<dyn NodeExtension>),
    }
}
";
    let slots = child_slots(declaration);
    assert!(
        slots.fields.contains("arms"),
        "Fix: the parser missed a declared child body: {slots:?}"
    );
    assert!(
        !slots.fields.contains("guard") && !slots.fields.contains("var"),
        "Fix: the parser took an operand for a child body: {slots:?}"
    );
    assert_eq!(
        slots.tuple_variants,
        BTreeSet::from(["Block".to_string()]),
        "Fix: an opaque extension is not a child body and a tuple body is"
    );

    let descent = "\
fn walk(node: &Node) -> bool {
    match node {
        Node::Speculate { arms, .. } => arms.iter().any(walk),
        _ => false,
    }
}
";
    assert_eq!(
        blocks_in(descent, &slots).len(),
        1,
        "Fix: a hand-written descent over a newly declared child body is not reported, so the \
         roster below cannot see it"
    );
    assert!(
        blocks_in(descent, &ChildSlots::default()).is_empty(),
        "Fix: the scanner reports without a vocabulary, so it is not the declaration it reads"
    );

    let renamed = "\
fn walk(node: &Node) -> bool {
    match node {
        Node::Speculate { arms: taken, .. } => taken.iter().any(walk),
        _ => false,
    }
}
";
    assert_eq!(
        blocks_in(renamed, &slots).len(),
        1,
        "Fix: a pattern that renames the field it binds escapes the scan"
    );

    let tuple = "\
fn walk(node: &Node) -> bool {
    match node {
        Node::Block(inner) => inner.iter().any(walk),
        _ => false,
    }
}
";
    assert_eq!(
        blocks_in(tuple, &slots).len(),
        1,
        "Fix: a tuple body binds under whatever the pattern calls it, and the scan has to follow \
         the binder rather than a field name"
    );

    let constructed = "\
fn rebuild(node: &Node) -> Node {
    match node {
        Node::Return => Node::Block(vec![]),
        _ => {
            let rebuilt = collect(node);
            Node::Block(rebuilt)
        }
    }
}
";
    assert!(
        blocks_in(constructed, &slots).is_empty(),
        "Fix: constructing a node is not destructuring one, and reporting a rebuild that never \
         reads a child body is how the previous ratchet died"
    );
}

/// Every reported block in the tree, ordered.
fn scan(root: &Path) -> Vec<Site> {
    let slots = declared_child_slots(root);
    let mut sites = Vec::new();
    for (relative, text) in rust_sources_with_text(root) {
        for line in blocks_in(&text, &slots) {
            sites.push(Site {
                path: relative.clone(),
                line,
            });
        }
    }
    sites.sort();
    sites
}

/// The child slots `Node` declares, read from the enum in the tree at `root`.
///
/// # Panics
///
/// Panics when the enum cannot be read or declares no child slot at all. A
/// scanner with an empty vocabulary reports nothing and every assertion built on
/// it passes, so a vocabulary that came back empty is a failure and not a clean
/// tree.
fn declared_child_slots(root: &Path) -> ChildSlots {
    let path = root.join(NODE_ENUM_SOURCE);
    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the `Node` enum is the scanner's vocabulary and {} could not be read: {error}. \
             If the enum moved, point NODE_ENUM_SOURCE at its new home.",
            path.display()
        )
    });
    let slots = child_slots(&text);
    assert!(
        !slots.fields.is_empty() && !slots.tuple_variants.is_empty(),
        "Fix: no child slot was read out of {}, so the scan below measures nothing. `Node` holds \
         named body fields and at least one tuple body, and the parser found fields {:?} and \
         tuple variants {:?}.",
        path.display(),
        slots.fields,
        slots.tuple_variants
    );
    slots
}

/// Opening line of every reported block in `text`.
fn blocks_in(text: &str, slots: &ChildSlots) -> Vec<usize> {
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
        let binders = child_body_binders(body, slots);
        if has_top_level_wildcard_arm(body)
            && recurses_on_a_binder(body, &binders)
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

/// The child slots `Node` declares.
///
/// A named field is reported by its own name because a pattern binds it by that
/// name. A tuple body has no field name, so the variant is recorded instead and
/// the binder is read off the pattern at the site.
#[derive(Debug, Default, PartialEq, Eq)]
struct ChildSlots {
    fields: BTreeSet<String>,
    tuple_variants: BTreeSet<String>,
}

/// The child slots declared by the `Node` enum in `text`.
fn child_slots(text: &str) -> ChildSlots {
    let source = mask_comments_and_strings(text);
    let mut slots = ChildSlots::default();
    let Some(open) = enum_body_open(&source, "Node") else {
        return slots;
    };
    let Some(end) = matching_brace(source.as_bytes(), open) else {
        return slots;
    };
    for variant in top_level_parts(&source[open + 1..end]) {
        let variant = variant.trim();
        let name_length = variant
            .bytes()
            .take_while(|byte| is_word_byte(*byte))
            .count();
        let (name, rest) = variant.split_at(name_length);
        let rest = rest.trim_start();
        if let Some(fields) = delimited(rest, b'{', b'}') {
            for field in top_level_parts(fields) {
                let Some((field_name, kind)) = field.split_once(':') else {
                    continue;
                };
                if holds_a_node(kind) {
                    slots.fields.insert(field_name.trim().to_string());
                }
            }
        } else if let Some(kinds) = delimited(rest, b'(', b')') {
            if top_level_parts(kinds).into_iter().any(holds_a_node) {
                slots.tuple_variants.insert(name.to_string());
            }
        }
    }
    slots
}

/// Offset of the `{` opening the body of `enum_name`, else `None`.
fn enum_body_open(source: &str, enum_name: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut cursor = 0;
    while let Some(offset) = source[cursor..].find(enum_name) {
        let start = cursor + offset;
        cursor = start + enum_name.len();
        let before_is_word = start
            .checked_sub(1)
            .is_some_and(|previous| is_word_byte(bytes[previous]));
        if before_is_word {
            continue;
        }
        let tail = &source[cursor..];
        let head = tail.trim_start();
        if head.starts_with('{') {
            return Some(cursor + (tail.len() - head.len()));
        }
    }
    None
}

/// The inside of `text` when it opens with `open`, else `None`.
fn delimited(text: &str, open: u8, close: u8) -> Option<&str> {
    if text.as_bytes().first() != Some(&open) {
        return None;
    }
    let end = matching_delimiter(text.as_bytes(), 0, open, close)?;
    Some(&text[1..end])
}

/// Index of the delimiter closing the one at `at`.
fn matching_delimiter(bytes: &[u8], at: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(at) {
        if *byte == open {
            depth += 1;
        } else if *byte == close {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

/// `text` split on the commas that sit outside every bracket.
///
/// A type argument list carries its own commas, so splitting on every comma
/// reads `Vec<Node>` as two fields and loses the field it was reading.
fn top_level_parts(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (index, byte) in text.bytes().enumerate() {
        match byte {
            b'{' | b'(' | b'[' | b'<' => depth += 1,
            b'}' | b')' | b']' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect()
}

/// Whether a declared type holds `Node` values.
///
/// Matched as a whole word, so `Arc<dyn NodeExtension>` is not a child body. An
/// extension node hides its own contents behind a trait object and the owner
/// walks report it as opaque rather than as a body slot.
fn holds_a_node(kind: &str) -> bool {
    let bytes = kind.as_bytes();
    let mut cursor = 0;
    while let Some(offset) = kind[cursor..].find("Node") {
        let start = cursor + offset;
        cursor = start + "Node".len();
        let before_is_word = start
            .checked_sub(1)
            .is_some_and(|previous| is_word_byte(bytes[previous]));
        let after_is_word = bytes.get(cursor).copied().is_some_and(is_word_byte);
        if !before_is_word && !after_is_word {
            return true;
        }
    }
    false
}

/// Every name a `Node::` pattern in `body` binds a child body to.
///
/// A named field binds under its own name unless the pattern renames it, and a
/// tuple body binds under whatever the pattern calls it, which is why the tuple
/// variants are carried separately: `Node::Block(nodes)` and
/// `Node::Block(inner)` are the same descent under two binder names.
fn child_body_binders(body: &str, slots: &ChildSlots) -> BTreeSet<String> {
    let mut binders = BTreeSet::new();
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find("Node::") {
        let start = cursor + offset;
        cursor = start + "Node::".len();
        let rest = &body[cursor..];
        let name_length = rest.bytes().take_while(|byte| is_word_byte(*byte)).count();
        let (variant, tail) = rest.split_at(name_length);
        let head = tail.trim_start();
        let inside = match head.as_bytes().first() {
            Some(b'{') => delimited(head, b'{', b'}'),
            Some(b'(') => delimited(head, b'(', b')'),
            _ => None,
        };
        let Some(inside) = inside else {
            continue;
        };
        if !is_a_pattern(&head[inside.len() + 2..]) {
            continue;
        }
        if head.starts_with('{') {
            for field in top_level_parts(inside) {
                let (name, renamed) = match field.split_once(':') {
                    Some((name, renamed)) => (name.trim(), Some(renamed)),
                    None => (field.trim(), None),
                };
                if !slots.fields.contains(name) {
                    continue;
                }
                let bound = renamed.map_or(name, |renamed| renamed.trim());
                if let Some(binder) = binder_name(bound) {
                    binders.insert(binder);
                }
            }
        } else if slots.tuple_variants.contains(variant) {
            for position in top_level_parts(inside) {
                if let Some(binder) = binder_name(position) {
                    binders.insert(binder);
                }
            }
        }
    }
    binders
}

/// Whether what follows a `Node::` form makes it a pattern rather than a value.
///
/// A pattern is followed by the arrow, another alternative, or a guard; a
/// constructed node is followed by the punctuation of the expression it sits in.
/// Without the distinction a rebuild that constructs `Node::Block(rebuilt)` and
/// iterates `rebuilt` reads as a hand-written descent.
fn is_a_pattern(after: &str) -> bool {
    let after = after.trim_start();
    after.starts_with("=>")
        || after.starts_with('|')
        || after.starts_with(')')
        || after.starts_with("if ")
}

/// The identifier a pattern position binds, if it binds one.
fn binder_name(position: &str) -> Option<String> {
    let name = position
        .trim()
        .trim_start_matches("ref ")
        .trim_start_matches("mut ")
        .trim_start_matches('&')
        .trim();
    if name.is_empty() || name == "_" || name == ".." {
        return None;
    }
    if !name.bytes().all(is_word_byte) {
        return None;
    }
    Some(name.to_string())
}

/// True when `body` iterates a bound child body or hands it to a call.
fn recurses_on_a_binder(body: &str, binders: &BTreeSet<String>) -> bool {
    binders.iter().any(|binder| {
        let iterated = [
            format!("{binder}.iter()"),
            format!("{binder}.iter_mut()"),
            format!("in {binder}"),
            format!("in &{binder}"),
            format!("in {binder}.iter()"),
            format!("({binder})"),
            format!("(&{binder})"),
            format!("({binder},"),
            format!("(&{binder},"),
        ];
        iterated.iter().any(|form| body.contains(form.as_str()))
    })
}
