//! Tier 2.5 line-index  -  write a per-byte line number into `lines[i]`.
//!
//! Every parser dialect that reports diagnostics needs line numbers.
//! This op is a GPU-native flag/scan/finalize pipeline. The first pass
//! marks bytes that terminate a line, the reduce substrate computes an
//! inclusive prefix sum of those marks, and the final pass writes the
//! line number for every byte position.
//!
//! Carriage-return handling: `\r` alone (Mac classic), `\r\n` (Windows),
//! and bare `\n` (Unix) are all normalized  -  `\r` does NOT increment
//! the counter (the following `\n` does), and a `\r` not followed by
//! `\n` increments on the `\r` itself. This matches `str::lines()`
//! semantics for byte-counting purposes.
//!
//! Ranged use: `column_for_byte(idx)` is `idx - line_start_offset`.
//! This primitive deliberately publishes per-byte line numbers only;
//! dialects that need column offsets derive them from their own
//! line-start representation.

use std::sync::Arc;

use crate::reduce::multi_block_prefix_scan::{multi_block_prefix_scan_sum_u32, BLOCK_LANES};
use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Stable op id for the registered Tier 3 wrapper.
pub const OP_ID: &str = "vyre-primitives::text::line_index";
const FLAG_OP_ID: &str = "vyre-primitives::text::line_index::break_flags";
const FINALIZE_OP_ID: &str = "vyre-primitives::text::line_index::finalize";

/// Build a Program that writes `lines[i] = line_number_of(source[i])`.
///
/// Newline bytes belong to the line they terminate, so the generated
/// pipeline computes an inclusive prefix of per-byte line-break flags
/// and subtracts the current byte's own flag before storing `lines[i]`.
#[must_use]
pub fn line_index(source: &str, lines: &str, n: u32) -> Program {
    match try_line_index(source, lines, n) {
        Ok(program) => program,
        Err(error) => crate::invalid_output_program(OP_ID, lines, DataType::U32, error),
    }
}

fn try_line_index(source: &str, lines: &str, n: u32) -> Result<Program, String> {
    if n == 0 {
        return Ok(empty_line_index_program(source, lines));
    }

    let flags = format!("__{lines}_line_break_flags");
    let prefix = format!("__{lines}_line_break_prefix");

    let flag_pass = line_break_flags_program(source, &flags, n)?;
    let scan_pass = multi_block_prefix_scan_sum_u32(&flags, &prefix, n);
    if scan_pass.stats().trap() {
        return Err(format!(
            "line_index n={n} could not build its prefix-scan pass. Fix: shard the source before line indexing or repair reduce::multi_block_prefix_scan sizing."
        ));
    }
    let finalize_pass = line_index_finalize_program(&flags, &prefix, lines, n)?;

    vyre_foundation::execution_plan::fusion::fuse_programs(&[flag_pass, scan_pass, finalize_pass])
        .map(|program| demote_intermediate_outputs(program, lines))
        .map_err(|error| {
            format!(
                "line_index fusion failed for n={n}: {error}. Fix: repair flag/scan/finalize fusion instead of falling back to a serial lane-0 loop."
            )
        })
}

fn empty_line_index_program(source: &str, lines: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(source, 0, BufferAccess::ReadOnly, DataType::U32).with_count(0),
            BufferDecl::output(lines, 1, DataType::U32)
                .with_count(0)
                .with_output_byte_range(0..0),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(Vec::new()),
        }],
    )
}

fn output_byte_range(words: u32, context: &str) -> Result<usize, String> {
    usize::try_from(words)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            format!(
                "{context} words={words} overflows output byte range. Fix: shard the source before GPU line indexing."
            )
        })
}

fn line_break_flags_program(source: &str, flags: &str, n: u32) -> Result<Program, String> {
    let t = Expr::InvocationId { axis: 0 };
    let next_idx = Expr::add(t.clone(), Expr::u32(1));
    let output_bytes = output_byte_range(n, "line_index break-flags output")?;

    let lane_body = vec![
        Node::let_bind(
            "byte",
            Expr::bitand(Expr::load(source, t.clone()), Expr::u32(0xFF)),
        ),
        Node::let_bind("next_byte", Expr::u32(0)),
        Node::if_then(
            Expr::lt(next_idx.clone(), Expr::u32(n)),
            vec![Node::assign(
                "next_byte",
                Expr::bitand(Expr::load(source, next_idx), Expr::u32(0xFF)),
            )],
        ),
        Node::let_bind("flag", Expr::u32(0)),
        Node::if_then(
            Expr::eq(Expr::var("byte"), Expr::u32(0x0A)),
            vec![Node::assign("flag", Expr::u32(1))],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("byte"), Expr::u32(0x0D)),
                Expr::and(
                    Expr::lt(Expr::add(t.clone(), Expr::u32(1)), Expr::u32(n)),
                    Expr::ne(Expr::var("next_byte"), Expr::u32(0x0A)),
                ),
            ),
            vec![Node::assign("flag", Expr::u32(1))],
        ),
        Node::store(flags, t.clone(), Expr::var("flag")),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(source, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(flags, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n)
                .with_pipeline_live_out(true)
                .with_output_byte_range(0..output_bytes),
        ],
        [BLOCK_LANES, 1, 1],
        vec![Node::Region {
            generator: Ident::from(FLAG_OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(Expr::lt(t, Expr::u32(n)), lane_body)]),
        }],
    ))
}

fn line_index_finalize_program(
    flags: &str,
    prefix: &str,
    lines: &str,
    n: u32,
) -> Result<Program, String> {
    let t = Expr::InvocationId { axis: 0 };
    let output_bytes = output_byte_range(n, "line_index lines output")?;
    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(n)),
        vec![Node::store(
            lines,
            t.clone(),
            Expr::sub(Expr::load(prefix, t.clone()), Expr::load(flags, t)),
        )],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(flags, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(prefix, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(lines, 2, DataType::U32)
                .with_count(n)
                .with_output_byte_range(0..output_bytes),
        ],
        [BLOCK_LANES, 1, 1],
        vec![Node::Region {
            generator: Ident::from(FINALIZE_OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    ))
}

fn demote_intermediate_outputs(program: Program, final_output: &str) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .map(|buffer| {
            let mut buffer = buffer.clone();
            if buffer.name() != final_output && buffer.is_output() {
                buffer.is_output = false;
                buffer.pipeline_live_out = true;
            }
            buffer
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}

/// Reference oracle: same line-counting semantics as the GPU kernel.
#[must_use]
#[cfg(any(test, feature = "cpu-parity", feature = "text"))]
pub fn reference_line_index(source: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(source.len());
    let mut line: u32 = 0;
    let mut prev_was_cr = false;
    for &byte in source {
        // Lone `\r` (not followed by `\n`) means the current byte
        // belongs to the next line  -  increment BEFORE recording this
        // byte's line number.
        if prev_was_cr && byte != b'\n' {
            line += 1;
        }
        out.push(line);
        if byte == b'\n' {
            line += 1;
            prev_was_cr = false;
        } else {
            prev_was_cr = byte == b'\r';
        }
    }
    out
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || line_index("source", "lines", 5),
        Some(|| {
            vec![vec![
                vec![0x61, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00],
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_no_newlines() {
        assert_eq!(reference_line_index(b"Hello"), vec![0; 5]);
    }

    #[test]
    fn reference_unix_lf() {
        // "ab\ncd" → lines [0, 0, 0, 1, 1]
        assert_eq!(reference_line_index(b"ab\ncd"), vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn reference_windows_crlf() {
        // "ab\r\ncd" → lines [0, 0, 0, 0, 1, 1]
        assert_eq!(reference_line_index(b"ab\r\ncd"), vec![0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn reference_mac_classic_cr() {
        // "ab\rcd" → lines [0, 0, 0, 1, 1]
        assert_eq!(reference_line_index(b"ab\rcd"), vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn reference_multiple_newlines() {
        // "a\n\nb" → lines [0, 0, 1, 2]
        assert_eq!(reference_line_index(b"a\n\nb"), vec![0, 0, 1, 2]);
    }

    #[test]
    fn reference_trailing_lone_cr_does_not_increment_after_eof() {
        // "ab\r" → lines [0, 0, 0]; we don't see a follow-up byte.
        assert_eq!(reference_line_index(b"ab\r"), vec![0, 0, 0]);
    }

    #[test]
    fn builder_uses_parallel_scan_pipeline() {
        let program = line_index("source", "lines", BLOCK_LANES + 17);
        assert_eq!(program.workgroup_size(), [BLOCK_LANES, 1, 1]);
        assert!(program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "__lines_line_break_flags"
                && buffer.is_pipeline_live_out()));
        assert!(program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "__lines_line_break_prefix"
                && buffer.is_pipeline_live_out()));
        assert_eq!(
            program
                .buffers()
                .iter()
                .filter(|buffer| buffer.is_output())
                .count(),
            1
        );
    }
}
