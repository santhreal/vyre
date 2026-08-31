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
    let mut body = vec![
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
    ];
    body.extend(ast_cse_hash_probe_body(
        ast_opcodes,
        ast_vals,
        hash_set,
        hash_set_capacity,
        out_modified_flag,
        t,
        Expr::var("h"),
        Expr::rem(Expr::var("h"), Expr::u32(hash_set_capacity)),
        SlotAccess::Contended,
    ));
    vec![Node::if_then(
        Expr::or(
            Expr::eq(Expr::var("op"), Expr::u32(AST_ADD)),
            Expr::eq(Expr::var("op"), Expr::u32(AST_PTR_DEREF)),
        ),
        body,
    )]
}

/// Build the standalone structural-hash CSE primitive.
#[must_use]
pub fn ast_cse_structural_hash_program(num_nodes: u32, hash_set_capacity: u32) -> Program {
    let t = Expr::LogicalIndex { axis: 0 };
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
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
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
                                    SlotAccess::SingleLane,
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

/// How a lane claims and reads a probe slot.
///
/// The dedup wave runs one lane per AST node and races for slots, so it claims
/// with compare-exchange and reads the winner's index atomically. The standalone
/// probe runs under `LogicalIndex == 0`, where a plain load and store observe
/// the same order and an atomic would only buy a fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotAccess {
    /// Every lane in the dispatch races for the same table.
    Contended,
    /// One lane owns the table for the whole probe.
    SingleLane,
}

impl SlotAccess {
    /// Read the slot's hash word, claiming it for `h` when it is free.
    ///
    /// `Contended` claims and reads in one compare-exchange; `SingleLane` reads
    /// and leaves the claim to [`Self::claim_nodes`].
    fn read_hash(self, hash_set: &str, slot_hash: Expr, h: Expr) -> Expr {
        match self {
            Self::Contended => Expr::atomic_compare_exchange(hash_set, slot_hash, Expr::u32(0), h),
            Self::SingleLane => Expr::load(hash_set, slot_hash),
        }
    }

    /// Finish claiming a free slot for `node_idx`.
    fn claim_nodes(
        self,
        hash_set: &str,
        slot_hash: Expr,
        slot_idx: Expr,
        node_idx: Expr,
        h: Expr,
    ) -> Vec<Node> {
        match self {
            // The compare-exchange already wrote the hash word.
            Self::Contended => vec![Node::let_bind(
                "_",
                Expr::atomic_exchange(hash_set, slot_idx, node_idx),
            )],
            Self::SingleLane => vec![
                Node::store(hash_set, slot_hash, h),
                Node::store(hash_set, slot_idx, node_idx),
            ],
        }
    }

    /// Read the index the slot's winner recorded.
    fn read_index(self, hash_set: &str, slot_idx: Expr) -> Expr {
        match self {
            Self::Contended => Expr::atomic_add(hash_set, slot_idx, Expr::u32(0)),
            Self::SingleLane => Expr::load(hash_set, slot_idx),
        }
    }
}

/// Body of the hash-table probe and deduplication step.
///
/// One bounded linear probe over a table of `(hash, earliest index)` pairs: the
/// lane claims the first free slot for its own hash, or finds its hash already
/// there and rewrites its node into a reference to the earlier one. Both the
/// contended dedup wave and the standalone probe walk exactly this, and each
/// carried its own copy until the pair drifted in the order of the equal-hash
/// branch.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ast_cse_hash_probe_body(
    ast_opcodes: &str,
    ast_vals: &str,
    hash_set: &str,
    hash_set_capacity: u32,
    out_modified_flag: &str,
    node_idx: Expr,
    h: Expr,
    slot_init: Expr,
    access: SlotAccess,
) -> Vec<Node> {
    let slot_hash = || Expr::var("slot_hash");
    let slot_idx = || Expr::var("slot_idx");
    let probe_h = || Expr::var("probe_h");
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
                    Node::let_bind("slot_idx", Expr::add(slot_hash(), Expr::u32(1))),
                    Node::let_bind(
                        "old_hash",
                        access.read_hash(hash_set, slot_hash(), probe_h()),
                    ),
                    Node::if_then(Expr::eq(Expr::var("old_hash"), Expr::u32(0)), {
                        let mut claimed = access.claim_nodes(
                            hash_set,
                            slot_hash(),
                            slot_idx(),
                            node_idx.clone(),
                            probe_h(),
                        );
                        claimed.push(Node::assign("active", Expr::bool(false)));
                        claimed
                    }),
                    Node::if_then(
                        Expr::eq(Expr::var("old_hash"), probe_h()),
                        vec![
                            Node::let_bind("earliest", access.read_index(hash_set, slot_idx())),
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
        SlotAccess::SingleLane,
    );
    let guarded = vec![Node::if_then(
        Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
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
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
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
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
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
