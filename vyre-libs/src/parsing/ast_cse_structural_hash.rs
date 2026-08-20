//! Structural-hash CSE probe/insert wave.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::ast_ops::{AST_ADD, AST_PTR_DEREF, AST_VAR};
use crate::hash::fnv1a::fnv1a32_mul_xor_word_expr;

/// Stable op id for the structural CSE child region.
pub const OP_ID: &str = "vyre-libs::parsing::ast_cse_structural_hash";

/// Stable op id for the hash-table probe and deduplication step.
pub const AST_CSE_HASH_PROBE_OP_ID: &str = "vyre-libs::parsing::ast_cse_hash_probe";
/// Emit the structural-hash deduplication phase.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ast_cse_structural_hash(
    ast_opcodes: &str,
    ast_lefts: &str,
    ast_rights: &str,
    ast_vals: &str,
    hash_set: &str,
    hash_set_capacity: u32,
    out_modified_flag: &str,
    t: Expr,
) -> Vec<Node> {
    vec![Node::if_then(
        Expr::or(
            Expr::eq(Expr::var("op"), Expr::u32(AST_ADD)),
            Expr::eq(Expr::var("op"), Expr::u32(AST_PTR_DEREF)),
        ),
        vec![
            Node::let_bind("l_idx2", Expr::load(ast_lefts, t.clone())),
            Node::let_bind("r_idx2", Expr::load(ast_rights, t.clone())),
            Node::let_bind(
                "h",
                fnv1a32_mul_xor_word_expr(Expr::var("op"), Expr::var("l_idx2")),
            ),
            Node::assign(
                "h",
                fnv1a32_mul_xor_word_expr(Expr::var("h"), Expr::var("r_idx2")),
            ),
            Node::let_bind(
                "slot",
                Expr::rem(Expr::var("h"), Expr::u32(hash_set_capacity)),
            ),
            Node::let_bind("active", Expr::bool(true)),
            Node::loop_for(
                "probe",
                Expr::u32(0),
                Expr::u32(hash_set_capacity),
                vec![Node::if_then(
                    Expr::var("active"),
                    vec![
                        Node::let_bind("slot_hash", Expr::mul(Expr::var("slot"), Expr::u32(2))),
                        Node::let_bind("slot_idx", Expr::add(Expr::var("slot_hash"), Expr::u32(1))),
                        Node::let_bind(
                            "old_hash",
                            Expr::atomic_compare_exchange(
                                hash_set,
                                Expr::var("slot_hash"),
                                Expr::u32(0),
                                Expr::var("h"),
                            ),
                        ),
                        Node::if_then(
                            Expr::eq(Expr::var("old_hash"), Expr::u32(0)),
                            vec![
                                Node::let_bind(
                                    "_",
                                    Expr::atomic_exchange(
                                        hash_set,
                                        Expr::var("slot_idx"),
                                        t.clone(),
                                    ),
                                ),
                                Node::assign("active", Expr::bool(false)),
                            ],
                        ),
                        Node::let_bind(
                            "earliest",
                            Expr::Select {
                                cond: Box::new(Expr::eq(Expr::var("old_hash"), Expr::var("h"))),
                                true_val: Box::new(Expr::atomic_add(
                                    hash_set,
                                    Expr::var("slot_idx"),
                                    Expr::u32(0),
                                )),
                                false_val: Box::new(Expr::u32(u32::MAX)),
                            },
                        ),
                        Node::if_then(
                            Expr::and(
                                Expr::eq(Expr::var("old_hash"), Expr::var("h")),
                                Expr::lt(Expr::var("earliest"), t.clone()),
                            ),
                            vec![
                                Node::store(ast_opcodes, t.clone(), Expr::u32(AST_VAR)),
                                Node::store(ast_vals, t.clone(), Expr::var("earliest")),
                                Node::let_bind(
                                    "_",
                                    Expr::atomic_add(out_modified_flag, Expr::u32(0), Expr::u32(1)),
                                ),
                            ],
                        ),
                        Node::if_then(
                            Expr::eq(Expr::var("old_hash"), Expr::var("h")),
                            vec![Node::assign("active", Expr::bool(false))],
                        ),
                        Node::assign(
                            "slot",
                            Expr::rem(
                                Expr::add(Expr::var("slot"), Expr::u32(1)),
                                Expr::u32(hash_set_capacity),
                            ),
                        ),
                    ],
                )],
            ),
        ],
    )]
}

/// Build the standalone structural-hash CSE primitive.
#[must_use]
pub fn ast_cse_structural_hash_program(num_nodes: u32, hash_set_capacity: u32) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let body = vec![
        Node::if_then(
            Expr::lt(t.clone(), Expr::u32(hash_set_capacity)),
            vec![
                Node::store("hash_set", Expr::mul(t.clone(), Expr::u32(2)), Expr::u32(0)),
                Node::store(
                    "hash_set",
                    Expr::add(Expr::mul(t.clone(), Expr::u32(2)), Expr::u32(1)),
                    Expr::u32(u32::MAX),
                ),
            ],
        ),
        Node::barrier(),
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![Node::loop_for(
                "node_idx",
                Expr::u32(0),
                Expr::u32(num_nodes),
                vec![
                    Node::let_bind("op", Expr::load("ast_opcodes", Expr::var("node_idx"))),
                    Node::if_then(
                        Expr::or(
                            Expr::eq(Expr::var("op"), Expr::u32(AST_ADD)),
                            Expr::eq(Expr::var("op"), Expr::u32(AST_PTR_DEREF)),
                        ),
                        vec![
                            Node::let_bind(
                                "l_idx2",
                                Expr::load("ast_lefts", Expr::var("node_idx")),
                            ),
                            Node::let_bind(
                                "r_idx2",
                                Expr::load("ast_rights", Expr::var("node_idx")),
                            ),
                            Node::let_bind(
                                "h",
                                fnv1a32_mul_xor_word_expr(Expr::var("op"), Expr::var("l_idx2")),
                            ),
                            Node::assign(
                                "h",
                                fnv1a32_mul_xor_word_expr(Expr::var("h"), Expr::var("r_idx2")),
                            ),
                            wrap_child_region(
                                AST_CSE_HASH_PROBE_OP_ID,
                                Ident::from(OP_ID),
                                ast_cse_hash_probe_body(
                                    "ast_opcodes",
                                    "ast_vals",
                                    "hash_set",
                                    hash_set_capacity,
                                    "out_modified_flag",
                                    Expr::var("node_idx"),
                                    Expr::var("h"),
                                    Expr::rem(Expr::var("h"), Expr::u32(hash_set_capacity)),
                                ),
                            ),
                        ],
                    ),
                ],
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("ast_opcodes", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("ast_lefts", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("ast_rights", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("ast_vals", 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("hash_set", 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(hash_set_capacity.saturating_mul(2)),
            BufferDecl::storage(
                "out_modified_flag",
                5,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(1),
        ],
        [num_nodes.max(hash_set_capacity).max(1), 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Body of the hash-table probe and deduplication step.
#[must_use]
pub fn ast_cse_hash_probe_body(
    ast_opcodes: &str,
    ast_vals: &str,
    hash_set: &str,
    hash_set_capacity: u32,
    out_modified_flag: &str,
    node_idx: Expr,
    h: Expr,
    slot_init: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("probe_h", h),
        Node::let_bind("slot", slot_init),
        Node::let_bind("active", Expr::bool(true)),
        Node::loop_for(
            "probe",
            Expr::u32(0),
            Expr::u32(hash_set_capacity),
            vec![Node::if_then(
                Expr::var("active"),
                vec![
                    Node::let_bind("slot_hash", Expr::mul(Expr::var("slot"), Expr::u32(2))),
                    Node::let_bind("slot_idx", Expr::add(Expr::var("slot_hash"), Expr::u32(1))),
                    Node::let_bind("old_hash", Expr::load(hash_set, Expr::var("slot_hash"))),
                    Node::if_then(
                        Expr::eq(Expr::var("old_hash"), Expr::u32(0)),
                        vec![
                            Node::store(hash_set, Expr::var("slot_hash"), Expr::var("probe_h")),
                            Node::store(hash_set, Expr::var("slot_idx"), node_idx.clone()),
                            Node::assign("active", Expr::bool(false)),
                        ],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("old_hash"), Expr::var("probe_h")),
                        vec![
                            Node::let_bind("earliest", Expr::load(hash_set, Expr::var("slot_idx"))),
                            Node::if_then(
                                Expr::lt(Expr::var("earliest"), node_idx.clone()),
                                vec![
                                    Node::store(ast_opcodes, node_idx.clone(), Expr::u32(AST_VAR)),
                                    Node::store(ast_vals, node_idx, Expr::var("earliest")),
                                    Node::let_bind(
                                        "_",
                                        Expr::atomic_add(
                                            out_modified_flag,
                                            Expr::u32(0),
                                            Expr::u32(1),
                                        ),
                                    ),
                                ],
                            ),
                            Node::assign("active", Expr::bool(false)),
                        ],
                    ),
                    Node::assign(
                        "slot",
                        Expr::rem(
                            Expr::add(Expr::var("slot"), Expr::u32(1)),
                            Expr::u32(hash_set_capacity),
                        ),
                    ),
                ],
            )],
        ),
    ]
}

/// Build the standalone probe operation.
#[must_use]
pub fn ast_cse_hash_probe_program(hash_set_capacity: u32) -> Program {
    let body = ast_cse_hash_probe_body(
        "ast_opcodes",
        "ast_vals",
        "hash_set",
        hash_set_capacity,
        "out_modified_flag",
        Expr::u32(0),
        Expr::u32(100),
        Expr::u32(0),
    );
    let guarded = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        body,
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("ast_opcodes", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage("ast_vals", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage("hash_set", 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(hash_set_capacity.saturating_mul(2)),
            BufferDecl::storage(
                "out_modified_flag",
                3,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(AST_CSE_HASH_PROBE_OP_ID, guarded)],
    )
}

/// The probe leaves the opcode and value buffers as it found them, writes the
/// structural hash of the single `AST_ADD` node into the first table slot, and
/// reports no collision.
const EXPECTED_AST_CSE_HASH_PROBE_OPS_BYTES: [u8; 4] = [10, 0, 0, 0];
const EXPECTED_AST_CSE_HASH_PROBE_VALS_BYTES: [u8; 4] = [0, 0, 0, 0];
const EXPECTED_AST_CSE_HASH_PROBE_TABLE_BYTES: [u8; 64] = {
    let mut bytes = [0u8; 64];
    bytes[0] = 100;
    bytes
};
const EXPECTED_AST_CSE_HASH_PROBE_COUNT_BYTES: [u8; 4] = [0, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        AST_CSE_HASH_PROBE_OP_ID,
        || ast_cse_hash_probe_program(8),
        Some(|| vec![vec![
            fixture_u32(&[AST_ADD]),
            fixture_u32(&[0]),
            fixture_u32(&[0; 16]),
            fixture_u32(&[0]),
        ]]),
        Some(|| {
            vec![vec![
                EXPECTED_AST_CSE_HASH_PROBE_OPS_BYTES.to_vec(),
                EXPECTED_AST_CSE_HASH_PROBE_VALS_BYTES.to_vec(),
                EXPECTED_AST_CSE_HASH_PROBE_TABLE_BYTES.to_vec(),
                EXPECTED_AST_CSE_HASH_PROBE_COUNT_BYTES.to_vec(),
            ]]
        }),
    )
}

fn fixture_u32(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

const EXPECTED_AST_CSE_STRUCTURAL_HASH_OPCODES_BYTES: [u8; 8] = [10, 0, 0, 0, 1, 0, 0, 0];
const EXPECTED_AST_CSE_STRUCTURAL_HASH_VALS_BYTES: [u8; 8] = [0; 8];
const EXPECTED_AST_CSE_STRUCTURAL_HASH_SET_BYTES: [u8; 64] = [
    0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
    0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
    0, 0, 0, 0, 255, 255, 255, 255, 175, 201, 24, 125, 0, 0, 0, 0,
];
const EXPECTED_AST_CSE_STRUCTURAL_HASH_MODIFIED_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || ast_cse_structural_hash_program(2, 8),
        Some(|| vec![vec![
            fixture_u32(&[AST_ADD, AST_ADD]),
            fixture_u32(&[1, 1]),
            fixture_u32(&[2, 2]),
            fixture_u32(&[0, 0]),
            fixture_u32(&[0; 16]),
            fixture_u32(&[0]),
        ]]),
        Some(|| vec![vec![
            EXPECTED_AST_CSE_STRUCTURAL_HASH_OPCODES_BYTES.to_vec(),
            EXPECTED_AST_CSE_STRUCTURAL_HASH_VALS_BYTES.to_vec(),
            EXPECTED_AST_CSE_STRUCTURAL_HASH_SET_BYTES.to_vec(),
            EXPECTED_AST_CSE_STRUCTURAL_HASH_MODIFIED_BYTES.to_vec(),
        ]]),
    )
}
