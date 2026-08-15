//! The three per-row phases of typedef annotation, as ops of their own.
//!
//! `c11_annotate_typedef_names` used to carry every phase inline, which put it
//! at 613 statement nodes against a 200 budget, control-flow depth 20 against
//! 6, and 37 loops against 8. The composition-discipline gate has no exemption
//! list by design, so the op was red.
//!
//! Each phase here answers one question about one VAST row and returns a
//! single `u32`. The annotator calls them and the calls inline away before
//! lowering, so the emitted kernel is unchanged while the op the gate measures
//! is a fraction of its former size.
//!
//! A callee may not read `InvocationId`, so the row index arrives as an
//! argument. The VAST node table and the source haystack arrive as buffer
//! references, which inlining retargets onto the caller's own buffers.

use super::super::phase_witness::PHASE_WITNESS_ROWS;
use super::*;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::operation::OperationFixtures;

#[cfg(any(test, feature = "cpu-parity"))]
use super::super::phase_witness::PhaseWitness;
#[cfg(any(test, feature = "cpu-parity"))]
use super::super::ref_typedef::{declaration_kind_at, scope_open_before, visible_declaration_kind};

/// Scope-open row for a given row, from the reverse scope walk.
pub(super) const SCOPE_OPEN_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_scope_open_for_row";
pub(super) use super::super::build::{
    DECL_KIND_FOR_ROW_OP_ID, DECL_KIND_FOR_ROW_PACKED_OP_ID, VISIBLE_NAME_FOR_ROW_OP_ID,
    VISIBLE_NAME_FOR_ROW_PACKED_OP_ID,
};

/// Callee-local buffer names. These never survive inlining: a buffer argument
/// retargets onto the caller's buffer and a scalar argument is substituted.
const NODES: &str = "phase_vast_nodes";
const HAYSTACK: &str = "phase_haystack";
const ROW: &str = "phase_row";
const HAYSTACK_LEN: &str = "phase_haystack_len";
const NUM_NODES: &str = "phase_num_nodes";
const RESULT: &str = "phase_result";
const SCOPE_OPEN_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: NODES,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: ROW,
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: RESULT,
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

const HAYSTACK_PHASE_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: NODES,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: HAYSTACK,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: ROW,
            ty: "u32",
        },
        TypedParam {
            name: HAYSTACK_LEN,
            ty: "u32",
        },
        TypedParam {
            name: NUM_NODES,
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: RESULT,
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

/// Rows a callee's own buffer declarations are sized for.
///
/// Sized for the registered witness so its fixture node buffer fits the
/// declared shape. This is still only a declaration extent: a callee is never
/// dispatched on its own, so this is the shape the registry validates against,
/// and inlining replaces every access with one on the caller's buffer, which
/// carries the real extent.
const PHASE_DECL_ROWS: u32 = PHASE_WITNESS_ROWS;

/// The row index the callee works on, read from its scalar parameter.
fn row() -> Expr {
    Expr::load(ROW, Expr::u32(0))
}

/// Assemble a phase program: `body` computes `out_name`, which becomes the
/// op's single output.
///
/// `with_haystack` controls whether the haystack and its length are declared.
/// The scope walk reads neither, and declaring parameters it never uses would
/// force every caller to pass two dead arguments.
fn phase_program(op_id: &str, with_haystack: bool, out_name: &str, mut body: Vec<Node>) -> Program {
    let mut buffers = vec![
        BufferDecl::storage(NODES, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(PHASE_DECL_ROWS.saturating_mul(VAST_NODE_STRIDE_U32)),
    ];
    let out_binding = if with_haystack {
        buffers.push(
            BufferDecl::storage(HAYSTACK, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        );
        buffers
            .push(BufferDecl::storage(ROW, 2, BufferAccess::ReadOnly, DataType::U32).with_count(1));
        buffers.push(
            BufferDecl::storage(HAYSTACK_LEN, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
        );
        buffers.push(
            BufferDecl::storage(NUM_NODES, 4, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        );
        5
    } else {
        buffers
            .push(BufferDecl::storage(ROW, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1));
        2
    };
    buffers.push(BufferDecl::output(RESULT, out_binding, DataType::U32).with_count(1));

    body.push(Node::store(RESULT, Expr::u32(0), Expr::var(out_name)));
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
    .with_entry_op_id(op_id)
}

/// Walk backwards from `row` to the innermost enclosing `{` that is still
/// open, and return its row index, or `SENTINEL` when the row is at file
/// scope.
///
/// This runs for EVERY row, not only identifiers: the CPU oracle writes
/// `scope_open_before(node_idx)` to the scope field unconditionally, so gating
/// it on `raw_kind == TOK_IDENTIFIER` leaves the carrier at `SENTINEL` on
/// every brace, paren and semicolon and diverges on all of them.
pub(super) fn c11_typedef_scope_open_for_row() -> Program {
    let t = row();
    let body = vec![
        Node::let_bind("scope_open", Expr::u32(SENTINEL)),
        Node::let_bind("scope_depth", Expr::u32(0)),
        Node::loop_for(
            "scope_scan",
            Expr::u32(0),
            t.clone(),
            vec![
                Node::let_bind(
                    "scope_rev",
                    Expr::sub(Expr::sub(t, Expr::u32(1)), Expr::var("scope_scan")),
                ),
                Node::let_bind(
                    "scope_kind",
                    Expr::load(
                        NODES,
                        Expr::mul(Expr::var("scope_rev"), Expr::u32(VAST_NODE_STRIDE_U32)),
                    ),
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("scope_open"), Expr::u32(SENTINEL)),
                        Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_RBRACE)),
                    ),
                    vec![Node::assign(
                        "scope_depth",
                        Expr::add(Expr::var("scope_depth"), Expr::u32(1)),
                    )],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("scope_open"), Expr::u32(SENTINEL)),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_LBRACE)),
                        vec![Node::if_then_else(
                            Expr::eq(Expr::var("scope_depth"), Expr::u32(0)),
                            vec![Node::assign("scope_open", Expr::var("scope_rev"))],
                            vec![Node::assign(
                                "scope_depth",
                                Expr::sub(Expr::var("scope_depth"), Expr::u32(1)),
                            )],
                        )],
                    )],
                ),
            ],
        ),
    ];
    phase_program(SCOPE_OPEN_FOR_ROW_OP_ID, false, "scope_open", body)
}

/// Row the scope-walk and declaration-scan fixtures ask about: the declarator
/// `v` inside the block. Both scans walk backwards from it and stop on real
/// structure rather than falling out at their initial sentinel.
#[cfg(any(test, feature = "cpu-parity"))]
const WITNESS_DECLARATOR_ROW: u32 = 11;

/// Row the visibility fixtures ask about: the use of the typedef name `T`
/// inside the block, which resolves back to the file-scope `typedef int T`.
#[cfg(any(test, feature = "cpu-parity"))]
const WITNESS_TYPEDEF_USE_ROW: u32 = 10;

#[cfg(any(test, feature = "cpu-parity"))]
fn scope_open_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    vec![vec![
        witness.node_bytes,
        WITNESS_DECLARATOR_ROW.to_le_bytes().to_vec(),
    ]]
}

#[cfg(any(test, feature = "cpu-parity"))]
fn scope_open_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let expected = scope_open_before(&witness.node_words, WITNESS_DECLARATOR_ROW as usize);
    vec![vec![expected.to_le_bytes().to_vec()]]
}

/// Buffers for `HAYSTACK_PHASE_SIGNATURE`, in declaration order.
#[cfg(any(test, feature = "cpu-parity"))]
fn haystack_phase_witness_inputs(row: u32, packed_haystack: bool) -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let haystack = witness.haystack_bytes(packed_haystack);
    vec![vec![
        witness.node_bytes,
        haystack,
        row.to_le_bytes().to_vec(),
        (witness.source.len() as u32).to_le_bytes().to_vec(),
        PHASE_WITNESS_ROWS.to_le_bytes().to_vec(),
    ]]
}

#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, false)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_packed_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_TYPEDEF_USE_ROW, true)
}

/// The phase answers the visibility question as a flag, so the oracle's
/// three-valued declaration kind collapses to "is a visible typedef name".
#[cfg(any(test, feature = "cpu-parity"))]
fn visible_name_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let kind = visible_declaration_kind(
        &witness.node_words,
        WITNESS_TYPEDEF_USE_ROW as usize,
        &witness.source,
        witness.lexeme(WITNESS_TYPEDEF_USE_ROW),
    );
    let expected = u32::from(kind == 1);
    vec![vec![expected.to_le_bytes().to_vec()]]
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_DECLARATOR_ROW, false)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_packed_inputs() -> Vec<Vec<Vec<u8>>> {
    haystack_phase_witness_inputs(WITNESS_DECLARATOR_ROW, true)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn decl_kind_witness_expected() -> Vec<Vec<Vec<u8>>> {
    let witness = PhaseWitness::build();
    let expected = declaration_kind_at(
        &witness.node_words,
        WITNESS_DECLARATOR_ROW as usize,
        &witness.source,
    );
    vec![vec![expected.to_le_bytes().to_vec()]]
}

/// Bind one fixture to a cfg-selected const so `inventory::submit!` stays a
/// single const expression. The oracles behind these fixtures are themselves
/// gated on `cpu-parity`, so every other configuration registers `None`.
/// `default` implies `matching-dfa` implies `cpu-parity`, so every conformance
/// harness still sees them.
macro_rules! witness_fixture {
    ($name:ident = $function:ident) => {
        #[cfg(any(test, feature = "cpu-parity"))]
        const $name: Option<OperationFixtures> = Some($function);
        #[cfg(not(any(test, feature = "cpu-parity")))]
        const $name: Option<OperationFixtures> = None;
    };
}

witness_fixture!(SCOPE_OPEN_INPUTS = scope_open_witness_inputs);
witness_fixture!(SCOPE_OPEN_EXPECTED = scope_open_witness_expected);
witness_fixture!(VISIBLE_NAME_INPUTS = visible_name_witness_inputs);
witness_fixture!(VISIBLE_NAME_PACKED_INPUTS = visible_name_witness_packed_inputs);
witness_fixture!(VISIBLE_NAME_EXPECTED = visible_name_witness_expected);
witness_fixture!(DECL_KIND_INPUTS = decl_kind_witness_inputs);
witness_fixture!(DECL_KIND_PACKED_INPUTS = decl_kind_witness_packed_inputs);
witness_fixture!(DECL_KIND_EXPECTED = decl_kind_witness_expected);

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SCOPE_OPEN_FOR_ROW_OP_ID,
        c11_typedef_scope_open_for_row,
        SCOPE_OPEN_INPUTS,
        SCOPE_OPEN_EXPECTED,
    )
    .with_signature(SCOPE_OPEN_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        VISIBLE_NAME_FOR_ROW_OP_ID,
        super::super::build::c11_typedef_visible_name_for_row,
        VISIBLE_NAME_INPUTS,
        VISIBLE_NAME_EXPECTED,
    )
    .with_signature(HAYSTACK_PHASE_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        VISIBLE_NAME_FOR_ROW_PACKED_OP_ID,
        super::super::build::c11_typedef_visible_name_for_row_packed_haystack,
        VISIBLE_NAME_PACKED_INPUTS,
        VISIBLE_NAME_EXPECTED,
    )
    .with_signature(HAYSTACK_PHASE_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DECL_KIND_FOR_ROW_OP_ID,
        super::super::build::c11_typedef_decl_kind_for_row,
        DECL_KIND_INPUTS,
        DECL_KIND_EXPECTED,
    )
    .with_signature(HAYSTACK_PHASE_SIGNATURE)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DECL_KIND_FOR_ROW_PACKED_OP_ID,
        super::super::build::c11_typedef_decl_kind_for_row_packed_haystack,
        DECL_KIND_PACKED_INPUTS,
        DECL_KIND_EXPECTED,
    )
    .with_signature(HAYSTACK_PHASE_SIGNATURE)
}
