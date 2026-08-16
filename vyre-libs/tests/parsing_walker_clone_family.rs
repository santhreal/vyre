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
//! a human decides the new IR is correct and re-pins the affected constant.
//!
//! Three constants were re-recorded for two such changes. The two `shunting`
//! entries moved when every child region that had named itself by suffixing its
//! parent operation id took the `anonymous::` prefix instead: a phase boundary
//! inside one operation has no operation to name it with, and an audit reading
//! such a name as an id was demanding a registration for a building block that
//! must not exist. The `python/decorators` entry moved when the dotted-name walk
//! gained one owner: the decorator copy was missing the
//! `cursor != INVALID_POS` guard, which is the defect the collapse fixed.

#![cfg(feature = "parsing")]
#![forbid(unsafe_code)]

mod harness;

use harness::ir_fingerprint::assert_pinned_ir_fingerprints;
use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_libs::parsing::core::ast::shunting::{ast_shunting_yard, ast_shunting_yard_with_capacity};
use vyre_libs::parsing::go::parse::structure::{
    go_extract_declarations, go_extract_packages_and_imports,
};
use vyre_libs::parsing::python::parse::calls::python312_extract_calls;
use vyre_libs::parsing::python::parse::decorators::python312_extract_decorators;
use vyre_libs::parsing::python::parse::structure::{
    python312_extract_imports, python312_extract_structure, python312_extract_with_blocks,
};

const TOKENS: u32 = 16;

// ---------------------------------------------------------------------------
// IR tree navigation
// ---------------------------------------------------------------------------

/// Every `Node::Loop` in `nodes` whose induction variable is `var`, in
/// depth-first order.
fn loops<'a>(nodes: &'a [Node], var: &str, out: &mut Vec<&'a Node>) {
    for node in nodes {
        match node {
            Node::Loop {
                var: name, body, ..
            } => {
                if name.as_str() == var {
                    out.push(node);
                }
                loops(body, var, out);
            }
            Node::If {
                then, otherwise, ..
            } => {
                loops(then, var, out);
                loops(otherwise, var, out);
            }
            Node::Block(children) => loops(children, var, out),
            Node::Region { body, .. } => loops(body, var, out),
            _ => {}
        }
    }
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
    for node in nodes {
        match node {
            Node::Let { name: bound, .. } => {
                if bound.as_str() == name {
                    out.push(node);
                }
            }
            Node::If {
                then, otherwise, ..
            } => {
                lets(then, name, out);
                lets(otherwise, name, out);
            }
            Node::Loop { body, .. } => lets(body, name, out),
            Node::Block(children) => lets(children, name, out),
            Node::Region { body, .. } => lets(body, name, out),
            _ => {}
        }
    }
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
        .filter(|node| match node {
            Node::Loop { body, .. } => assignments_to(body, "brace_done"),
            _ => false,
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
    nodes.iter().any(|node| match node {
        Node::Assign { name, .. } => name.as_str() == target,
        Node::If {
            then, otherwise, ..
        } => assignments_to(then, target) || assignments_to(otherwise, target),
        Node::Loop { body, .. } => assignments_to(body, target),
        Node::Block(children) => assignments_to(children, target),
        Node::Region { body, .. } => assignments_to(body, target),
        _ => false,
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

/// Canonical wire fingerprints for every entry point the clone-family merges
/// pass through. Recorded on the pre-merge tree except for `python/decorators`,
/// whose pre-merge value encoded the drift the merge resolved: the missing
/// `cursor != INVALID_POS` guard in the decorator dotted-name walk.
const EXPECTED: &[(&str, &str)] = &[
    (
        "python/structure",
        "7b737f3c6d347e5d931914b094d83f1baf97cedcd03d0c91ea5a0e9aafbe3f2e",
    ),
    (
        "python/imports",
        "639ca8ff90ef863b10abdfcead55c657712717787a77c449dd1911f25596d0ed",
    ),
    (
        "python/with_blocks",
        "e5988ebad66407b2aa161e72ed30541541798e547f43b0b74a1178c68ccf92c8",
    ),
    (
        "python/calls",
        "b7fc4b37f8edeb2a21b8fef987255beb033406dcc783050d5e3fd560722fb7fd",
    ),
    (
        "python/decorators",
        "17f23ec1fd4cb40da1ef4f932e3c35deefbe71f088042f0d094a99efd4c90ba0",
    ),
    (
        "go/packages_and_imports",
        "3969eca2030b300e567181e6fa0e76dac573a88d3282011e2a7e86261a4dc69b",
    ),
    (
        "go/declarations",
        "f2995d3054afdee134f7ba19d85f3810d2cd3a223733f8b4b025c9c8773038ea",
    ),
    (
        "core/ast/shunting",
        "7d48e9cf92a5244fe5252e20108c8a2b8461577e152c934e3f5ee69f3c8a8c43",
    ),
    (
        "core/ast/shunting_with_capacity",
        "15008181950586f054720b52b712a14b34aa83ab10fb56f2bc871d49b5eca7df",
    ),
];

#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    assert_pinned_ir_fingerprints(&entry_points(), EXPECTED);
}

// ---------------------------------------------------------------------------
