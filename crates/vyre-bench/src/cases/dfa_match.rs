//! Full grid-strided word-aligned literal scan over 256KB of text.
//!
//! Scans a 256KB text buffer for every 4-byte-aligned occurrence of `b"vyre"`.
//! The grid is auto-inferred from the text buffer's element count (65536 u32
//! words), so every word is visited exactly once by a distinct thread.
//!
//! The CPU reference uses `memchr::memmem::find_iter` at byte granularity while
//! the GPU version aligns to u32 word boundaries. The needle is therefore
//! planted only at word-aligned offsets, so both count the same matches.

use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// 256K bytes / 4 bytes per u32.
const WORD_COUNT: u32 = 65_536;

pub(crate) static DFA_MATCH: MicroCase = MicroCase {
    id: "foundation.dfa_match.256k",
    name: "DFA Match 256K",
    summary: "Full-coverage word-aligned literal scan over 256K bytes with atomic match counting",
    tags: &["compute", "branching", "atomic"],
    contract: None,
    program,
    fixture,
    reference,
    work: MicroWork::Bytes {
        read: WORD_COUNT as u64 * 4,
        written: 4,
    },
};

fn program() -> Program {
    // The text buffer has 65536 elements. With workgroup [256,1,1], the driver
    // infers ceil(65536/256) = 256 workgroups, so 65536 total threads: each
    // thread checks exactly one word via gid_x().
    Program::wrapped(
        vec![
            BufferDecl::output("out_matches", 0, DataType::U32).with_count(1),
            BufferDecl::storage("text", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(WORD_COUNT),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("idx"), Expr::u32(WORD_COUNT)),
                    Expr::eq(
                        Expr::load("text", Expr::var("idx")),
                        Expr::u32(u32::from_le_bytes(*b"vyre")),
                    ),
                ),
                vec![Node::let_bind(
                    "_old",
                    Expr::atomic_add("out_matches", Expr::u32(0), Expr::u32(1)),
                )],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let size: usize = WORD_COUNT as usize * 4;
    let mut text_bytes = vec![0u8; size];
    for (index, byte) in text_bytes.iter_mut().enumerate() {
        *byte = b'a' + (index % 26) as u8;
    }
    // Word-aligned every 4096 bytes, so the byte-granularity CPU reference and
    // the word-granularity GPU scan agree on the count.
    for offset in (0..size.saturating_sub(4)).step_by(4096) {
        text_bytes[offset..offset + 4].copy_from_slice(b"vyre");
    }

    vec![text_bytes]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::dfa_vyre_match_count_bytes(
        &inputs[0],
    )]
}
