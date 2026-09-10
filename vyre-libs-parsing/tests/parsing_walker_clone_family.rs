//! One-owner guards for the `parsing/` token-walk clone families.
//!
//! Three families of duplicated builder code lived under
//! `vyre-libs/src/parsing/`: the Python dotted-name AST walk (four copies), the
//! Go brace-balanced span scan (two copies), and the `shunting`
//! file-plus-directory split (two registrations of one op id).
//!
//! Collapsing a clone family is only safe if the surviving owner emits exactly
//! what every former copy emitted. Each test below asserts that property
//! directly against the built IR, so a copy that gets reintroduced or a shared
//! helper that quietly grows a per-caller special case turns them red.
//!
//! What these do not catch: a deliberate IR change. That is the point at which
//! a human decides the new IR is correct and re-blesses the golden.
//!
//! # Why the pin is structural IR and not `Program::fingerprint`
//!
//! `clone_family_entry_points_emit_the_pinned_ir` compares each entry point's
//! canonicalized buffer roster and node tree against a checked-in golden.
//! `Program::fingerprint` is BLAKE3 over `canonical_wire_bytes`, and those
//! bytes open with `WIRE_FORMAT_VERSION`, so a serialization revision moves
//! every digest in the tree while no program's meaning moves, and the
//! difference is reported as 32 opaque bytes. The golden is a function of the
//! IR model, so a changed operand, a dropped node, a reordered data dependence
//! or a changed buffer moves it and a wire revision does not.

#![cfg(feature = "parsing")]
#![forbid(unsafe_code)]

use std::path::PathBuf;

use crate::harness;

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::visit::{any_descendant, child_bodies, for_each_node};
use vyre_libs_parsing::parsing::core::ast::shunting::{
    ast_shunting_yard, ast_shunting_yard_with_capacity,
};
use vyre_libs_parsing::parsing::go::parse::structure::{
    go_extract_declarations, go_extract_packages_and_imports,
};
use vyre_libs_parsing::parsing::python::parse::calls::python312_extract_calls;
use vyre_libs_parsing::parsing::python::parse::decorators::python312_extract_decorators;
use vyre_libs_parsing::parsing::python::parse::structure::{
    python312_extract_imports, python312_extract_structure, python312_extract_with_blocks,
};
use vyre_test_support::structural_ir::{
    golden_contains, render_structural_ir, write_golden, StructuralIrGolden,
};

const TOKENS: u32 = 16;

// ---------------------------------------------------------------------------
// IR tree navigation
// ---------------------------------------------------------------------------

/// Every `Node::Loop` in `nodes` whose induction variable is `var`, in
/// depth-first order.
fn loops<'a>(nodes: &'a [Node], var: &str, out: &mut Vec<&'a Node>) {
    for_each_node(nodes, |node| {
        if matches!(node, Node::Loop { var: name, .. } if name.as_str() == var) {
            out.push(node);
        }
    });
}

fn only_loop<'a>(program: &'a Program, var: &str) -> &'a Node {
    let mut found = Vec::new();
    loops(program.entry(), var, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one `{var}` loop in this program, found {}",
        found.len()
    );
    found[0]
}

fn all_loops<'a>(program: &'a Program, var: &str) -> Vec<&'a Node> {
    let mut found = Vec::new();
    loops(program.entry(), var, &mut found);
    found
}

/// Every `Node::Let` in `nodes` whose bound name is `name`, in depth-first
/// order.
fn lets<'a>(nodes: &'a [Node], name: &str, out: &mut Vec<&'a Node>) {
    for_each_node(nodes, |node| {
        if matches!(node, Node::Let { name: bound, .. } if bound.as_str() == name) {
            out.push(node);
        }
    });
}

fn only_let<'a>(program: &'a Program, name: &str) -> &'a Node {
    let mut found = Vec::new();
    lets(program.entry(), name, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one `let {name}` in this program, found {}",
        found.len()
    );
    found[0]
}

/// Rewrite the dotted-name walk's accumulator assignment to one canonical
/// name so the four extractors' walks can be compared directly. The
/// accumulator is the only thing the four copies are entitled to disagree on:
/// it is the caller's output variable, not part of the walk.
fn canonical_chain_accumulator(node: &Node) -> Node {
    match node {
        Node::Assign { name, value }
            if name.as_str() != "cursor"
                && matches!(value, Expr::Var(v) if v.as_str() == "after_dot") =>
        {
            Node::assign("chain_end", value.clone())
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => Node::If {
            cond: cond.clone(),
            then: then.iter().map(canonical_chain_accumulator).collect(),
            otherwise: otherwise.iter().map(canonical_chain_accumulator).collect(),
        },
        Node::Loop {
            var,
            from,
            to,
            body,
        } => Node::Loop {
            var: var.clone(),
            from: from.clone(),
            to: to.clone(),
            body: body.iter().map(canonical_chain_accumulator).collect(),
        },
        Node::Block(children) => {
            Node::Block(children.iter().map(canonical_chain_accumulator).collect())
        }
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn python_imports() -> Program {
    python312_extract_imports(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "out_records",
        "out_counts",
        TOKENS,
    )
}

fn python_with_blocks() -> Program {
    python312_extract_with_blocks(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "out_records",
        "out_counts",
        TOKENS,
    )
}

fn python_calls() -> Program {
    python312_extract_calls(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "out_calls",
        "out_call_counts",
        "out_kwargs",
        "out_kw_counts",
        TOKENS,
    )
}

fn python_decorators() -> Program {
    python312_extract_decorators(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "out_records",
        "out_counts",
        TOKENS,
    )
}

fn python_structure() -> Program {
    python312_extract_structure(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "out_records",
        "out_counts",
        TOKENS,
    )
}

fn go_declarations() -> Program {
    go_extract_declarations(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(TOKENS),
        "out_decls",
        "out_decl_counts",
    )
}

fn go_packages() -> Program {
    go_extract_packages_and_imports(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(TOKENS),
        "out_packages",
        "out_package_counts",
        "out_imports",
        "out_import_counts",
    )
}

fn shunting_default() -> Program {
    ast_shunting_yard(
        "tok_types",
        "statements",
        Expr::u32(100),
        "out_ast_nodes",
        "out_ast_count",
        "out_statement_roots",
        "scratch_val_stack",
        "scratch_op_stack",
    )
}

// ---------------------------------------------------------------------------
// Family 1: the Python dotted-name AST walk
// ---------------------------------------------------------------------------

/// `import a.b.c`, `with a.b.c() as x:`, `a.b.c(...)`, and `@a.b.c` all resolve
/// a dotted name with the same bounded segment walk. One owner must emit it.
///
/// The `cursor != INVALID_POS` guard in that walk is load-bearing: `cursor`
/// holds `u32::MAX` once the chain ends, `cursor + 1` wraps to 0, and an
/// unguarded rescan therefore restarts at token 0 and can pull an unrelated
/// `.ident` pair from the head of the unit into the accumulator.
#[test]
fn python_dotted_name_walk_has_one_owner() {
    let programs = [
        ("imports", python_imports()),
        ("with_blocks", python_with_blocks()),
        ("calls", python_calls()),
        ("decorators", python_decorators()),
    ];
    let walks: Vec<(&str, Node)> = programs
        .iter()
        .map(|(name, program)| {
            (
                *name,
                canonical_chain_accumulator(only_loop(program, "seg")),
            )
        })
        .collect();

    let (owner_name, owner) = &walks[0];
    for (name, walk) in &walks[1..] {
        assert_eq!(
            walk, owner,
            "the dotted-name walk in `{name}` is not the same walk as in `{owner_name}`"
        );
    }
}

/// The four carriers the walk assigns across its loop iterations must be
/// declared identically by every caller, or the walk reads a differently
/// initialized cursor.
#[test]
fn python_dotted_name_walk_carriers_have_one_owner() {
    for (name, program) in [
        ("imports", python_imports()),
        ("with_blocks", python_with_blocks()),
        ("calls", python_calls()),
        ("decorators", python_decorators()),
    ] {
        let dot_pos = only_let(&program, "dot_pos");
        let after_dot = only_let(&program, "after_dot");
        assert_eq!(
            dot_pos,
            &Node::let_bind("dot_pos", Expr::u32(u32::MAX)),
            "`{name}` seeds the walk's `dot_pos` carrier differently"
        );
        assert_eq!(
            after_dot,
            &Node::let_bind("after_dot", Expr::u32(u32::MAX)),
            "`{name}` seeds the walk's `after_dot` carrier differently"
        );
    }
}

// ---------------------------------------------------------------------------
// Family 2: the Go brace-balanced span scan
// ---------------------------------------------------------------------------

/// A Go function body and a Go interface body are both delimited by a
/// balanced brace pair, and `go_extract_declarations` scans for both. One
/// owner must emit that scan.
#[test]
fn go_brace_span_scan_has_one_owner() {
    let program = go_declarations();
    let scans = all_loops(&program, "scan");
    let brace_scans: Vec<&Node> = scans
        .iter()
        .copied()
        .filter(|node| {
            child_bodies(node)
                .into_iter()
                .any(|body| assignments_to(body, "brace_done"))
        })
        .collect();
    assert_eq!(
        brace_scans.len(),
        2,
        "expected the function-body and interface-body brace scans, found {}",
        brace_scans.len()
    );
    let bodies: Vec<&[Node]> = brace_scans
        .iter()
        .map(|node| match node {
            Node::Loop { body, .. } => body.as_slice(),
            _ => unreachable!("filtered to loops"),
        })
        .collect();
    assert_eq!(
        bodies[0], bodies[1],
        "the function-body and interface-body brace scans are not the same scan"
    );
}

fn assignments_to(nodes: &[Node], target: &str) -> bool {
    nodes.iter().any(|root| {
        any_descendant(root, &mut |node| {
            matches!(node, Node::Assign { name, .. } if name.as_str() == target)
        })
    })
}

// ---------------------------------------------------------------------------
// Family 3: the shunting module split
// ---------------------------------------------------------------------------

/// `core/ast/shunting.rs` and `core/ast/shunting/` both carried an
/// `inventory::submit!` for the shunting-yard op id, with different builders
/// and different expected outputs. Exactly one registration may reach the
/// registry.
#[test]
fn shunting_yard_has_one_registration() {
    let registrations = OperationRegistry::global()
        .iter()
        .filter(|operation| operation.id == "vyre-libs::parsing::ast_shunting_yard")
        .count();
    assert_eq!(
        registrations, 1,
        "the shunting-yard op must be registered exactly once"
    );
}

/// The capacity-bounded builder is the general form: given the default
/// capacities, it must emit what the default builder emits apart from the
/// buffer sizing the capacities exist to change.
#[test]
fn shunting_yard_capacity_form_shares_the_statement_pass() {
    let default = shunting_default();
    let bounded = ast_shunting_yard_with_capacity(
        "tok_types",
        "statements",
        Expr::u32(100),
        "out_ast_nodes",
        "out_ast_count",
        "out_statement_roots",
        "scratch_val_stack",
        "scratch_op_stack",
        65_536,
        100,
    );
    assert_eq!(
        only_loop(&default, "tok_idx"),
        only_loop(&bounded, "tok_idx"),
        "the per-statement token pass drifted between the two shunting-yard entry points"
    );
}

// ---------------------------------------------------------------------------
// Pinned IR for every entry point these merges touch
// ---------------------------------------------------------------------------

fn entry_points() -> Vec<(&'static str, Program)> {
    vec![
        ("python/structure", python_structure()),
        ("python/imports", python_imports()),
        ("python/with_blocks", python_with_blocks()),
        ("python/calls", python_calls()),
        ("python/decorators", python_decorators()),
        ("go/packages_and_imports", go_packages()),
        ("go/declarations", go_declarations()),
        ("core/ast/shunting", shunting_default()),
        (
            "core/ast/shunting_with_capacity",
            ast_shunting_yard_with_capacity(
                "tok_types",
                "statements",
                Expr::u32(100),
                "out_ast_nodes",
                "out_ast_count",
                "out_statement_roots",
                "scratch_val_stack",
                "scratch_op_stack",
                4_096,
                100,
            ),
        ),
    ]
}

/// Path of the structural IR golden.
fn golden_path() -> PathBuf {
    harness::crate_dir().join("tests/golden/parsing_walker_clone_family_ir.txt")
}

/// The structural IR golden for the entry points these merges touch.
fn golden() -> StructuralIrGolden {
    StructuralIrGolden::new(
        "vyre-libs-parsing/parsing-walker-clone-family-structural-ir/v1",
        "the parsing/ token-walk clone families",
        golden_path(),
    )
}

/// Every entry point's structural IR, rendered in golden order.
fn render_corpus() -> String {
    golden().render(entry_points())
}

/// The structural IR of every clone-family entry point, against the golden.
///
/// The golden carries the canonicalized buffer roster and node tree of each
/// entry point: node kinds, field names, operand expressions, literal values,
/// identifier text, region generators and nesting. A changed operand, a
/// dropped node, a reordered data dependence or a changed buffer moves it. A
/// wire format revision does not, because nothing here reads the wire
/// encoding. The single-owner properties the family collapse depends on are
/// asserted structurally by the tests above, so this rule reports drift and
/// does not carry that proof.
#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    golden().assert_matches(&render_corpus());
}

/// A golden that no longer names an entry point silently stopped covering it.
#[test]
fn the_golden_names_every_entry_point() {
    let corpus = std::fs::read_to_string(golden_path()).expect("structural IR golden must exist");
    for (id, _) in entry_points() {
        assert!(
            golden_contains(&corpus, id),
            "Fix: the structural IR golden is missing `{id}`; re-bless it."
        );
    }
}

/// The rendering must be a pure function of the program.
///
/// A renderer that read an address, a cache or an iteration order would match
/// the golden once and diverge on the next run, which reads as an IR change.
#[test]
fn structural_ir_is_deterministic_across_builds() {
    assert_eq!(render_corpus(), render_corpus());
}

#[test]
#[ignore = "bless: rewrites the pinned structural IR golden; run deliberately and review the diff"]
fn bless_pinned_structural_ir_golden() {
    golden().bless(&render_corpus());
}

/// Write the full structural IR of every entry point under the test target
/// directory, for reading a digest move the histograms do not explain.
///
/// The golden pins a digest over this rendering rather than the rendering
/// itself. This is how a maintainer gets the text: run it on both sides of the
/// change and diff the two trees.
#[test]
#[ignore = "diagnostic: writes the full structural IR rendering, for diffing a digest move"]
fn dump_full_structural_ir() {
    let out =
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("parsing_walker_structural_ir");
    for (id, program) in entry_points() {
        write_golden(
            &out.join(format!("{}.ir.txt", id.replace('/', "_"))),
            &render_structural_ir(&program),
        );
    }
    println!("wrote the full structural IR to {}", out.display());
}
