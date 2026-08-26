//! GPU-native CSE program builders for the encoded arena.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs::hash::fnv1a::{fnv1a32_initial_expr, fnv1a32_mix_word_expr};

use super::cse_via_encoded::WORKGROUP_X;
use super::expr_arena::expr_kind;

/// Build the structural-hash analysis Program. Single-workgroup,
/// fused level-loop with workgroup-scope barriers. Each thread
/// strides over expr ids in chunks of `WORKGROUP_X` per level.
///
/// Buffer layout:
///   0: arena_kinds (RO)
///   1: arena_arg0  (RO)
///   2: arena_arg1  (RO)
///   3: arena_arg2  (RO)
///   4: arena_depths (RO)
///   5: max_depth_buf (RO; single u32)
///   6: hash (RW; init zeros)
#[must_use]
pub fn build_structural_hash_program(expr_count: u32, max_depth_iter_cap: u32) -> Program {
    // Per-Expr body: structural-hash mixer. Critical invariant:
    // mix child HASHES (h0/h1/h2), never raw arg slots (a0/a1/a2)
    // for parent kinds  -  raw args carry arena-position-dependent
    // child ids that break canonical-equivalence across duplicates.
    // For leaves the raw a0/a1/a2 carry the actual payload (literal
    // value, name id, axis, buffer name id) and ARE structural.
    let mix = |var_name: &str| -> Vec<Node> {
        vec![Node::assign(
            "h",
            fnv1a32_mix_word_expr(Expr::var("h"), Expr::var(var_name)),
        )]
    };
    let per_expr_body = vec![
        Node::let_bind("kind", Expr::load("arena_kinds", Expr::var("i"))),
        Node::let_bind("a0", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind("a1", Expr::load("arena_arg1", Expr::var("i"))),
        Node::let_bind("a2", Expr::load("arena_arg2", Expr::var("i"))),
        // Child hashes (the post-order encoding guarantees children's
        // hashes are already written by the time the parent's level
        // runs). For leaves these reads are harmless (a0/a1/a2 carry
        // payloads that may index outside the arena, but `hash` was
        // zero-initialized so out-of-bounds reads return 0 inside the
        // backend's CSR-bounds clamp; the leaf branch ignores h0/h1/h2
        // anyway).
        Node::let_bind("h0", Expr::load("hash", Expr::var("a0"))),
        Node::let_bind("h1", Expr::load("hash", Expr::var("a1"))),
        Node::let_bind("h2", Expr::load("hash", Expr::var("a2"))),
        // Mix kind first (the family discriminator).
        Node::let_bind("h", fnv1a32_initial_expr()),
        Node::assign(
            "h",
            fnv1a32_mix_word_expr(Expr::var("h"), Expr::var("kind")),
        ),
        // Leaves with a payload in a0: literals, vars, buf_len,
        // invocation/workgroup/local id (axis lives in a0).
        Node::if_then(
            Expr::or(
                Expr::or(
                    Expr::or(
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LIT_U32)),
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LIT_I32)),
                    ),
                    Expr::or(
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LIT_F32)),
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LIT_BOOL)),
                    ),
                ),
                Expr::or(
                    Expr::or(
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::VAR)),
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BUF_LEN)),
                    ),
                    Expr::or(
                        Expr::or(
                            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::INVOCATION_ID)),
                            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::WORKGROUP_ID)),
                        ),
                        Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LOCAL_ID)),
                    ),
                ),
            ),
            mix("a0"),
        ),
        // BIN_OP: a0 = op_tag (structural), a1/a2 = child ids (NOT
        // structural). Mix op_tag + child hashes in position order.
        // (Commutative-friendly mixing was tried and reverted  -  the
        // extra Selects + tag-flag chain doubled the per-Expr kernel
        // runtime and the speculative CSE gain didn't justify it.)
        Node::if_then(Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BIN_OP)), {
            let mut body = mix("a0");
            body.extend(mix("h1"));
            body.extend(mix("h2"));
            body
        }),
        // UN_OP: a0 = op_tag, a1 = child id.
        Node::if_then(Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::UN_OP)), {
            let mut body = mix("a0");
            body.extend(mix("h1"));
            body
        }),
        // LOAD: a0 = buffer name id (structural), a1 = index Expr id.
        Node::if_then(Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::LOAD)), {
            let mut body = mix("a0");
            body.extend(mix("h1"));
            body
        }),
        // SELECT, FMA: 3 child ids in a0/a1/a2; payload-free.
        Node::if_then(
            Expr::or(
                Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::SELECT)),
                Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::FMA)),
            ),
            {
                let mut body = mix("h0");
                body.extend(mix("h1"));
                body.extend(mix("h2"));
                body
            },
        ),
        // SUBGROUP_LOCAL_ID / SUBGROUP_SIZE: payload-free; the kind
        // mix above is sufficient.
        Node::store("hash", Expr::var("i"), Expr::var("h")),
    ];

    super::arena_kernel::build_fused_level_wave_program(
        expr_count,
        max_depth_iter_cap,
        vec![BufferDecl::output("hash", 6, DataType::U32).with_count(expr_count.max(1))],
        per_expr_body,
    )
}

/// Build the canonical-id Program. Single dispatch: each thread `i`
/// computes `canonical[i]` by brute-force scanning `0..i` for the
/// smallest `j` that is structurally identical to `i`.
///
/// Structural identity requires BOTH the hash pre-filter AND a full
/// `(kind, arg0, arg1, arg2)` tuple comparison. The hash alone is a
/// 32-bit FNV value whose collision probability grows with arena size
/// (birthday bound ~0.3% per 5k-expr arena); relying on hash equality
/// alone would silently merge non-equivalent exprs (miscompile). The
/// tuple check is the definitive correctness guard; the hash serves
/// only as a fast-reject to reduce wasted tuple reads.
///
/// Buffer layout:
///   0: hash          (RO)
///   1: canonical     (output)
///   2: arena_kinds   (RO)
///   3: arena_arg0    (RO)
///   4: arena_arg1    (RO)
///   5: arena_arg2    (RO)
#[must_use]
pub fn build_canonical_id_program(expr_count: u32) -> Program {
    let mut buffers = vec![
        BufferDecl::storage("hash", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::output("canonical", 1, DataType::U32).with_count(expr_count.max(1)),
    ];
    // Structural tuple buffers: hash collision alone must never declare two
    // exprs equivalent. The four arena rows supply the definitive
    // (kind, arg0, arg1, arg2) tuple comparison.
    buffers.extend(super::arena_kernel::arena_row_buffers(expr_count, 2));

    // Per-thread body: brute-force scan 0..i.
    // The post-order encoding ensures children appear before parents,
    // so structurally-equivalent siblings always have a prior candidate
    // at a smaller index.
    //
    // Equivalence predicate: hash pre-filter (fast reject) THEN full
    // structural tuple check (correctness gate). Both must hold before
    // `found_canonical` is updated.
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("my_hash", Expr::load("hash", Expr::var("i"))),
                // Load this thread's structural tuple once (avoids
                // re-reading the same arena row on every inner iteration).
                Node::let_bind("my_kind", Expr::load("arena_kinds", Expr::var("i"))),
                Node::let_bind("my_a0", Expr::load("arena_arg0", Expr::var("i"))),
                // Child structural HASHES, not raw arg1/arg2. For BIN_OP /
                // UN_OP / LOAD the arg1/arg2 slots hold arena child indices,
                // which differ between structurally-equal duplicates that sit
                // at different positions; a raw index comparison would reject
                // those true duplicates (the hash mixer mixes child hashes for
                // exactly this reason). For every leaf kind arg1 = arg2 = 0, so
                // hash[0] == hash[0] holds trivially and leaf identity is
                // decided by `my_kind` + `my_a0` below.
                Node::let_bind(
                    "my_h1",
                    Expr::load("hash", Expr::load("arena_arg1", Expr::var("i"))),
                ),
                Node::let_bind(
                    "my_h2",
                    Expr::load("hash", Expr::load("arena_arg2", Expr::var("i"))),
                ),
                Node::let_bind("found_canonical", Expr::var("i")),
                Node::loop_for(
                    "j",
                    Expr::u32(0),
                    Expr::var("i"),
                    vec![
                        Node::let_bind("their_hash", Expr::load("hash", Expr::var("j"))),
                        // Gate 1: hash pre-filter. Mismatched hashes
                        // structurally different exprs almost always.
                        // Gate 2: structural confirmation. Hash equality
                        // alone is not structural identity because two
                        // distinct exprs can share a 32-bit hash value
                        // (birthday collision). Confirm kind, arg0 (op tag /
                        // payload), and BOTH child structural hashes match
                        // before declaring `j` canonical for `i`. The child
                        // hashes (not raw child indices) are what make
                        // position-independent duplicates compare equal.
                        // Gate 3: only take the first (smallest-index)
                        // match by checking `found_canonical == i`.
                        Node::if_then(
                            Expr::and(
                                Expr::and(
                                    Expr::and(
                                        Expr::and(
                                            Expr::and(
                                                Expr::eq(
                                                    Expr::var("their_hash"),
                                                    Expr::var("my_hash"),
                                                ),
                                                Expr::eq(
                                                    Expr::load("arena_kinds", Expr::var("j")),
                                                    Expr::var("my_kind"),
                                                ),
                                            ),
                                            Expr::eq(
                                                Expr::load("arena_arg0", Expr::var("j")),
                                                Expr::var("my_a0"),
                                            ),
                                        ),
                                        Expr::eq(
                                            Expr::load(
                                                "hash",
                                                Expr::load("arena_arg1", Expr::var("j")),
                                            ),
                                            Expr::var("my_h1"),
                                        ),
                                    ),
                                    Expr::eq(
                                        Expr::load(
                                            "hash",
                                            Expr::load("arena_arg2", Expr::var("j")),
                                        ),
                                        Expr::var("my_h2"),
                                    ),
                                ),
                                Expr::eq(Expr::var("found_canonical"), Expr::var("i")),
                            ),
                            vec![Node::assign("found_canonical", Expr::var("j"))],
                        ),
                    ],
                ),
                Node::store("canonical", Expr::var("i"), Expr::var("found_canonical")),
            ],
        ),
    ];

    Program::wrapped(buffers, [WORKGROUP_X, 1, 1], body)
}

/// Build a compact readback Program for CSE canonical ids.
///
/// Buffer layout:
///   0: canonical (RO)
///   1: canonical_delta (RW), where word 0 is an atomic pair count and
///      words `1 + 2*k .. 3 + 2*k` are `(expr_id, canonical_id)`.
#[must_use]
pub fn build_canonical_delta_compact_program(expr_count: u32) -> Program {
    let delta_words = expr_count.saturating_mul(2).saturating_add(1).max(1);
    let buffers = vec![
        BufferDecl::storage("canonical", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("canonical_delta", 1, BufferAccess::ReadWrite, DataType::U32)
            .with_count(delta_words),
    ];
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("canonical_id", Expr::load("canonical", Expr::var("i"))),
                Node::if_then(
                    Expr::ne(Expr::var("canonical_id"), Expr::var("i")),
                    vec![
                        Node::let_bind(
                            "slot",
                            Expr::atomic_add("canonical_delta", Expr::u32(0), Expr::u32(1)),
                        ),
                        Node::let_bind(
                            "base",
                            Expr::add(Expr::u32(1), Expr::mul(Expr::var("slot"), Expr::u32(2))),
                        ),
                        Node::store("canonical_delta", Expr::var("base"), Expr::var("i")),
                        Node::store(
                            "canonical_delta",
                            Expr::add(Expr::var("base"), Expr::u32(1)),
                            Expr::var("canonical_id"),
                        ),
                    ],
                ),
            ],
        ),
    ];

    Program::wrapped(buffers, [WORKGROUP_X, 1, 1], body)
}
