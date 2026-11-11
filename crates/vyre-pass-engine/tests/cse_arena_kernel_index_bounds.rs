//! Every index a CSE arena kernel derives from buffer data lands inside the
//! buffer it reads or writes.
//!
//! WHY: the encoded arena packs a child id and a leaf payload into the same
//! three word rows. A parent's `arg0` is a child index, a literal's `arg0` is
//! its value, and the hash kernel reads `hash[arg0]` for both. A payload
//! larger than the arena reads past the buffer, which a backend that bounds
//! nothing answers with whatever the address held. The same shape appears in
//! the tuple comparison of the canonical-id kernel and in the pair slot the
//! compaction kernel takes from an atomic counter.
//!
//! The cases drive each kernel with an arena whose every data word is far past
//! every extent, and with a compaction counter that starts full, so a kernel
//! that trusts a data word asks for an element no buffer has. The reference
//! interpreter refuses an out-of-range access, so the refusal is the assertion.
//!
//! What this does not catch: a fold that keeps the index in range but picks
//! the wrong element. Structural equivalence is judged by the CSE hoist tests,
//! which read the canonical ids these kernels produce.

#![cfg(feature = "optimizer")]

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_megakernel::{execute_single_program, Digest, SearchBudget, SemanticExecutionPolicy};
use vyre_pass_engine::optimizer::cse_via_encoded::{
    build_canonical_delta_compact_program, build_canonical_id_program,
    build_structural_hash_program,
};

/// Expr count every case builds its kernels for.
const EXPR_COUNT: u32 = 8;

/// A word no buffer in these kernels accepts as an index.
const HOSTILE_WORD: u32 = 0x7fff_ffff;

fn policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::unknown_policy(
        Digest([0; 32]),
        SearchBudget::new(8, 64, 0, 0, 1_000),
        1_000_000,
    )
}

/// `count` words of `word`, in the wire encoding the kernels read.
fn words(word: u32, count: u32) -> Vec<u8> {
    (0..count).flat_map(|_| word.to_le_bytes()).collect()
}

/// Execute `program` and fail with the access the kernel asked for.
fn executes(node_name: &str, program: vyre_foundation::ir::Program, inputs: &[Vec<u8>]) {
    execute_single_program(
        &ReferenceSemanticExecutor,
        node_name,
        program,
        inputs,
        &policy(),
    )
    .unwrap_or_else(|error| {
        panic!(
            "Fix: the {node_name} kernel must access only inside its buffers when every arena \
             word is {HOSTILE_WORD}. Fold each data-derived index with \
             `vyre_foundation::composition::bounded_index`, or gate the access with control \
             flow: {error:?}"
        )
    });
}

#[test]
fn the_structural_hash_kernel_reads_inside_the_hash_buffer() {
    // kinds, arg0, arg1, arg2, depths, max_depth.
    let inputs = vec![
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(0, EXPR_COUNT),
        words(1, 1),
    ];
    executes(
        "cse-structural-hash",
        build_structural_hash_program(EXPR_COUNT, 2),
        &inputs,
    );
}

#[test]
fn the_canonical_id_kernel_reads_inside_the_hash_buffer() {
    // hash, kinds, arg0, arg1, arg2.
    let inputs = vec![
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
        words(HOSTILE_WORD, EXPR_COUNT),
    ];
    executes(
        "cse-canonical-id",
        build_canonical_id_program(EXPR_COUNT),
        &inputs,
    );
}

#[test]
fn the_compaction_kernel_stores_inside_the_pair_buffer() {
    // Every expr reports a canonical other than itself, so every invocation
    // takes a slot, and the counter already stands at the last one.
    let canonical = words(HOSTILE_WORD, EXPR_COUNT);
    let mut delta = words(0, 2 * EXPR_COUNT + 1);
    delta[0..4].copy_from_slice(&EXPR_COUNT.to_le_bytes());
    executes(
        "cse-canonical-delta-compact",
        build_canonical_delta_compact_program(EXPR_COUNT),
        &[canonical, delta],
    );
}
