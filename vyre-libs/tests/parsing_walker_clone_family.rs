//! One-owner guards for the `parsing/` token-walk clone families.
//!
//! Five families of duplicated builder code lived under
//! `vyre-libs/src/parsing/`: the Python dotted-name AST walk (four copies), the
//! C lowering node-row builder (two copies), the C typedef identifier-hash pass
//! (two copies), the Go brace-balanced span scan (two copies), and the
//! `shunting` file-plus-directory split (two registrations of one op id).
//!
//! Collapsing a clone family is only safe if the surviving owner emits exactly
//! what every former copy emitted. Each test below asserts that property
//! directly against the built IR, so a copy that gets reintroduced or a shared
//! helper that quietly grows a per-caller special case turns them red.
//!
//! What these do not catch: a deliberate IR change. That is the point at which
//! a human decides the new IR is correct and re-pins the affected constant.

#![cfg(feature = "parsing")]
#![forbid(unsafe_code)]

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_libs::parsing::c::lower::ast_to_pg_nodes::{
    c_lower_ast_to_pg_semantic_graph, c_lower_ast_to_pg_semantic_graph_with_pg,
    c_lower_ast_to_pg_semantic_graph_with_pg_no_control_resolution,
    C_AST_PG_SEMANTIC_NODE_STRIDE_U32,
};
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_semantic_graph};
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_typedef_names, c11_annotate_typedef_names_packed_haystack,
    c11_annotate_typedef_names_precomputed_context,
    c11_annotate_typedef_names_precomputed_context_packed_haystack,
    c11_annotate_typedef_names_precomputed_scope,
    c11_annotate_typedef_names_precomputed_scope_packed_haystack, c11_prehash_vast_identifiers,
    c11_prehash_vast_identifiers_packed_haystack,
};
use vyre_libs::parsing::core::ast::shunting::{ast_shunting_yard, ast_shunting_yard_with_capacity};
use vyre_libs::parsing::go::parse::structure::{
    go_extract_declarations, go_extract_packages_and_imports,
};
use vyre_libs::parsing::python::parse::calls::python312_extract_calls;
use vyre_libs::parsing::python::parse::decorators::python312_extract_decorators;
use vyre_libs::parsing::python::parse::structure::{
    python312_extract_imports, python312_extract_structure, python312_extract_with_blocks,
};
use vyre_reference::value::Value;

const TOKENS: u32 = 16;
const VAST_STRIDE: usize = 10;
const SENTINEL: u32 = u32::MAX;
/// Semantic PG column holding the node's category.
const PG_CATEGORY_COLUMN: usize = 6;
/// Semantic PG column holding the node's role.
const PG_ROLE_COLUMN: usize = 7;

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

fn typedef_prehash() -> Program {
    c11_prehash_vast_identifiers(
        "vast_nodes",
        "haystack",
        Expr::u32(64),
        Expr::u32(8),
        "out_hashed_vast_nodes",
    )
}

fn typedef_annotate() -> Program {
    c11_annotate_typedef_names(
        "vast_nodes",
        "haystack",
        Expr::u32(64),
        Expr::u32(8),
        "out_annotated_vast_nodes",
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
// Family 2: the C lowering node-row builder
// ---------------------------------------------------------------------------

/// `c_lower_ast_to_pg_nodes` and the semantic-graph lowerer's plain-PG side
/// write the same six-column ProgramGraph row from the same six VAST row
/// fields. One builder must emit both.
///
/// Compared through `reference_eval` rather than through the IR tree because
/// the two write different buffers at different strides: what has to agree is
/// the row content, not the store expressions.
#[test]
fn c_lowering_pg_node_rows_have_one_owner() {
    let rows: Vec<[u32; VAST_STRIDE]> = (0..6u32)
        .map(|i| {
            [
                100 + i,
                if i == 0 { SENTINEL } else { 0 },
                if i == 0 { 1 } else { SENTINEL },
                if i + 1 < 6 { i + 1 } else { SENTINEL },
                SENTINEL,
                i * 4,
                3,
                i,
                i + 1,
                0,
            ]
        })
        .collect();
    let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let n = rows.len();

    let structural = c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(n as u32), "out_pg_nodes");
    let structural_rows = unpack(
        &vyre_reference::reference_eval(
            &structural,
            &[
                Value::from(pack(&flat)),
                Value::from(pack(&vec![0u32; n * 6])),
            ],
        )
        .expect("structural lowerer runs")[0]
            .to_bytes(),
    );

    let semantic = c_lower_ast_to_pg_semantic_graph_with_pg(
        "vast_nodes",
        Expr::u32(n as u32),
        "out_plain_pg_nodes",
        "out_pg_nodes",
        "out_pg_edges",
    );
    let semantic_outputs = vyre_reference::reference_eval(
        &semantic,
        &[
            Value::from(pack(&flat)),
            Value::from(pack(&vec![0u32; n * 6])),
            Value::from(pack(&vec![
                0u32;
                n * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize
            ])),
            Value::from(pack(&vec![0u32; n * 5 * 6])),
        ],
    )
    .expect("semantic lowerer runs");
    let plain_rows = unpack(&semantic_outputs[0].to_bytes());
    let semantic_rows = unpack(&semantic_outputs[1].to_bytes());

    assert!(
        structural_rows.iter().any(|&w| w != 0),
        "fixture must produce non-empty PG rows"
    );
    assert_eq!(
        structural_rows, plain_rows,
        "the semantic lowerer's plain-PG rows drifted from `c_lower_ast_to_pg_nodes`"
    );
    for row in 0..n {
        let structural_row = &structural_rows[row * 6..row * 6 + 6];
        let semantic_row = &semantic_rows[row * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize..]
            [..C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize];
        assert_eq!(
            structural_row,
            &semantic_row[..6],
            "row {row}: the semantic PG row's first six columns drifted from the structural row"
        );
    }
}

/// Every kind's `(category, role)` pair must agree between the GPU
/// classification chain and the CPU oracle table. The kind space is read out of
/// the parser's own constant file at run time, so a kind wired into one owner
/// and not the other goes red here rather than in a corpus months later.
///
/// The pinned classified count fails by default on any new kind that gets a
/// role: adding one requires recording the decision here.
#[test]
fn c_semantic_classification_has_one_owner() {
    let kinds = declared_vast_kinds();
    assert!(
        kinds.len() > 100,
        "expected the parser's full VAST kind set, scanned {}",
        kinds.len()
    );

    let flat: Vec<u32> = kinds
        .iter()
        .enumerate()
        .flat_map(|(i, &(_, kind))| {
            let i = i as u32;
            [kind, SENTINEL, SENTINEL, SENTINEL, SENTINEL, i, 1, i, 0, 0]
        })
        .collect();
    let n = kinds.len();
    let packed = pack(&flat);

    let program = c_lower_ast_to_pg_semantic_graph(
        "vast_nodes",
        Expr::u32(n as u32),
        "out_pg_nodes",
        "out_pg_edges",
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(packed.clone()),
            Value::from(pack(&vec![
                0u32;
                n * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize
            ])),
            Value::from(pack(&vec![0u32; n * 5 * 6])),
        ],
    )
    .expect("semantic lowerer runs over the full kind sweep");
    let gpu = unpack(&outputs[0].to_bytes());
    let oracle = unpack(&reference_ast_to_pg_semantic_graph(&packed).nodes);

    let stride = C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize;
    let mut disagreements = Vec::new();
    let mut classified = 0usize;
    for (row, (name, kind)) in kinds.iter().enumerate() {
        let gpu_category = gpu[row * stride + PG_CATEGORY_COLUMN];
        let gpu_role = gpu[row * stride + PG_ROLE_COLUMN];
        let oracle_category = oracle[row * stride + PG_CATEGORY_COLUMN];
        let oracle_role = oracle[row * stride + PG_ROLE_COLUMN];
        if gpu_role != 0 || gpu_category != 0 {
            classified += 1;
        }
        if (gpu_category, gpu_role) != (oracle_category, oracle_role) {
            disagreements.push(format!(
                "  {name} ({kind:#010x}): gpu=({gpu_category}, {gpu_role}) oracle=({oracle_category}, {oracle_role})"
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "the GPU classification chain and the CPU oracle table disagree on {} of {n} kinds:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
    assert_eq!(
        classified, CLASSIFIED_VAST_KINDS,
        "the number of VAST kinds carrying a semantic category or role changed. \
         Re-pin CLASSIFIED_VAST_KINDS only with a recorded decision for the new kind."
    );
}

/// Kinds that resolve to a non-zero category or role. Pinned so a kind added
/// to the classification surface cannot land silently.
const CLASSIFIED_VAST_KINDS: usize = 80;

/// Read `(name, value)` for every `C_AST_KIND_*` constant out of the parser's
/// own constant file, plus the shared predicate kind the tables also key on.
///
/// Scanned rather than listed so the sweep cannot go stale behind a new kind.
fn declared_vast_kinds() -> Vec<(String, u32)> {
    let source = std::fs::read_to_string(
        vyre_test_support::monorepo::vyre_workspace_root()
            .join("vyre-libs/src/parsing/c/parse/vast_kinds.rs"),
    )
    .expect("the parser's VAST kind constants must be readable");
    let mut kinds: Vec<(String, u32)> = source
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("pub const C_AST_KIND_")?;
            let (name, rest) = rest.split_once(": u32 = ")?;
            let literal = rest.trim_end_matches(';').trim().replace('_', "");
            let value = literal
                .strip_prefix("0x")
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| literal.parse::<u32>().ok())?;
            Some((format!("C_AST_KIND_{name}"), value))
        })
        .collect();
    kinds.push((
        "node_kind::FUNCTION_DECL".to_string(),
        vyre_primitives::predicate::node_kind::FUNCTION_DECL,
    ));
    kinds
}

// ---------------------------------------------------------------------------
// Family 3: the C typedef identifier-hash pass
// ---------------------------------------------------------------------------

/// `c11_prehash_vast_identifiers` and `c11_annotate_typedef_names` both read a
/// VAST row's identifier span and fold it into the same FNV-1a symbol hash.
/// One pass must emit that, or the two disagree on a symbol hash and the
/// typedef scope lookup silently misses.
///
/// The two are entitled to differ on when the fold runs: the annotator skips
/// rows the prehash pass already filled. That guard is the enclosing `if`, not
/// the fold, so it is deliberately outside what this compares.
#[test]
fn c_typedef_identifier_hash_has_one_owner() {
    let prehash = typedef_prehash();
    let annotate = typedef_annotate();

    assert_eq!(
        only_loop(&prehash, "hash_i"),
        only_loop(&annotate, "hash_i"),
        "the identifier FNV-1a fold drifted between the prehash and annotate passes"
    );
    for field in ["raw_kind", "tok_start", "tok_len", "name_hash"] {
        assert_eq!(
            only_let(&prehash, field),
            only_let(&annotate, field),
            "the `{field}` row binding drifted between the prehash and annotate passes"
        );
    }
}

// ---------------------------------------------------------------------------
// Family 4: the Go brace-balanced span scan
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
// Family 5: the shunting module split
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
        (
            "c/lower/pg_nodes",
            c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(8), "out_pg_nodes"),
        ),
        (
            "c/lower/semantic_graph",
            c_lower_ast_to_pg_semantic_graph(
                "vast_nodes",
                Expr::u32(8),
                "out_pg_nodes",
                "out_pg_edges",
            ),
        ),
        (
            "c/lower/semantic_graph_with_pg",
            c_lower_ast_to_pg_semantic_graph_with_pg(
                "vast_nodes",
                Expr::u32(8),
                "out_plain_pg_nodes",
                "out_pg_nodes",
                "out_pg_edges",
            ),
        ),
        (
            "c/lower/semantic_graph_no_control_resolution",
            c_lower_ast_to_pg_semantic_graph_with_pg_no_control_resolution(
                "vast_nodes",
                Expr::u32(8),
                "out_plain_pg_nodes",
                "out_pg_nodes",
                "out_pg_edges",
            ),
        ),
        ("c/typedef/prehash", typedef_prehash()),
        (
            "c/typedef/prehash_packed",
            c11_prehash_vast_identifiers_packed_haystack(
                "vast_nodes",
                "haystack",
                Expr::u32(64),
                Expr::u32(8),
                "out_hashed_vast_nodes",
            ),
        ),
        ("c/typedef/annotate", typedef_annotate()),
        (
            "c/typedef/annotate_packed",
            c11_annotate_typedef_names_packed_haystack(
                "vast_nodes",
                "haystack",
                Expr::u32(64),
                Expr::u32(8),
                "out_annotated_vast_nodes",
            ),
        ),
        (
            "c/typedef/annotate_precomputed_scope",
            c11_annotate_typedef_names_precomputed_scope(
                "vast_nodes",
                "haystack",
                Expr::u32(64),
                Expr::u32(8),
                "out_annotated_vast_nodes",
            ),
        ),
        (
            "c/typedef/annotate_precomputed_scope_packed",
            c11_annotate_typedef_names_precomputed_scope_packed_haystack(
                "vast_nodes",
                "haystack",
                Expr::u32(64),
                Expr::u32(8),
                "out_annotated_vast_nodes",
            ),
        ),
        (
            "c/typedef/annotate_precomputed_context",
            c11_annotate_typedef_names_precomputed_context(
                "vast_nodes",
                "haystack",
                "decl_contexts",
                "visible_type",
                Expr::u32(64),
                Expr::u32(8),
                "out_annotated_vast_nodes",
            ),
        ),
        (
            "c/typedef/annotate_precomputed_context_packed",
            c11_annotate_typedef_names_precomputed_context_packed_haystack(
                "vast_nodes",
                "haystack",
                "decl_contexts",
                "visible_type",
                Expr::u32(64),
                Expr::u32(8),
                "out_annotated_vast_nodes",
            ),
        ),
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
/// pass through. Recorded on the pre-merge tree except for `python/decorators`
/// and the three `c/lower/semantic_graph*` entry points, whose pre-merge values
/// encoded the two drifts the merge resolved: the missing
/// `cursor != INVALID_POS` guard in the decorator dotted-name walk, and the GPU
/// category chain that classified `C_AST_KIND_GNU_LOCAL_LABEL_DECL` as `NONE`
/// where the CPU oracle classified it as `GNU`.
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
        "c/lower/pg_nodes",
        "2c3848290c4142a9d2fb0f765b78962249cfa1c476cb6d6999034627fa0d9e94",
    ),
    (
        "c/lower/semantic_graph",
        "98064a8dfe2ea2f5e388c5844a7504bb88c323fe50ceb045a7e7f6e309a5bb89",
    ),
    (
        "c/lower/semantic_graph_with_pg",
        "d0d6de67afd0bdc58e643d5d3b25a0192b73e8ffad7a6010317669d3d66b207c",
    ),
    (
        "c/lower/semantic_graph_no_control_resolution",
        "dd858320f80fdcc74ee930a57f291f0d955c23a5dfb74994e7fdab9c044ac935",
    ),
    (
        "c/typedef/prehash",
        "67089650fe3d2fd1a3e1f9b6d7e9809827e6ec99e2860029290749da9acb765b",
    ),
    (
        "c/typedef/prehash_packed",
        "904c2fce2e924bf6094a18c0ef9c022b12c0a5aa021c4f5ab9ed2c19dfb84519",
    ),
    (
        "c/typedef/annotate",
        "6db939a6803bfdd3ff2737dce55148b286e7cc423c7fe25bb58878a3d5278c4c",
    ),
    (
        "c/typedef/annotate_packed",
        "18b7c9c68dfc0b0e13cb699ee07500328a10bf81ed7e758f6bb6309d5f0c16ce",
    ),
    (
        "c/typedef/annotate_precomputed_scope",
        "ae4669ac4bc674140aa83e739a653ae95ded1ba22abdc80a4e819deafd4e8a72",
    ),
    (
        "c/typedef/annotate_precomputed_scope_packed",
        "0f186e4b08f3ecf2da61a1d95238fb442abd697d0391f3e055d5c9e569094b44",
    ),
    (
        "c/typedef/annotate_precomputed_context",
        "abccd5a2d40f76baa7965ead0c16b32df00632dd730dfa425bde67d9a9c61e5d",
    ),
    (
        "c/typedef/annotate_precomputed_context_packed",
        "d3226140cc432183ffba55a3ed98caf8fe0f231530702456516ee4d00ee98e7c",
    ),
    (
        "core/ast/shunting",
        "964eddf2fd3f882eb0e873daadabb325ed4ad98fd1c945698c43d916e7a7411a",
    ),
    (
        "core/ast/shunting_with_capacity",
        "2df3aee68b2c9f8f5efb0f259d0a293a2ebc9636982e9cce9f0876da0a3f5054",
    ),
];

fn hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    let actual: Vec<(&'static str, String)> = entry_points()
        .iter()
        .map(|(name, program)| (*name, hex(program)))
        .collect();
    assert_eq!(
        actual.len(),
        EXPECTED.len(),
        "fixture count drifted from the pinned table"
    );
    let mut report = String::new();
    let mut drifted = false;
    for ((name, got), (pinned_name, pinned)) in actual.iter().zip(EXPECTED.iter()) {
        assert_eq!(name, pinned_name, "fixture order drifted from the table");
        if got != pinned {
            drifted = true;
        }
        report.push_str(&format!("    (\"{name}\", \"{got}\"),\n"));
    }
    assert!(
        !drifted,
        "generated IR changed for at least one clone-family entry point. \
         Recorded fingerprints:\n{report}"
    );
}

// ---------------------------------------------------------------------------

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}
