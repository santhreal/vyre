//! Line index  -  write a per-byte line number into `lines[i]`.
//!
//! Every parser dialect that reports diagnostics needs line numbers.
//! This op is a GPU-native flag/scan pipeline. The first pass marks bytes
//! where the line number increments from the previous byte, and the reduce
//! substrate scans those marks directly into the line number for every byte
//! position.
//!
//! Carriage-return handling: `\r` alone (Mac classic), `\r\n` (Windows),
//! and bare `\n` (Unix) are all normalized. Newline bytes belong to the
//! line they terminate; the following byte starts the next line when one
//! exists. This matches `str::lines()` semantics for byte-counting purposes.
//!
//! Ranged use: `column_for_byte(idx)` is `idx - line_start_offset`.
//! This primitive deliberately publishes per-byte line numbers only;
//! dialects that need column offsets derive them from their own
//! line-start representation.

use std::sync::Arc;
use vyre_foundation::composition::{tag_program, trap_program, wrap_anonymous_region};

use crate::reduce::multi_block_prefix_scan::multi_block_prefix_scan_sum_u32_with_block_lanes;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, PORTABLE_WORKGROUP_INVOCATIONS,
};
use vyre_foundation::GeometryRequirements;

/// Stable op id for the registered dialect wrapper.
pub const LINE_INDEX_OP_ID: &str = "vyre-libs::text::line_index";
const FLAG_OP_ID: &str = "anonymous::vyre-primitives::text::line_index::line_start_flags";

/// Return the execution geometry requirements for line indexing.
#[must_use]
pub const fn line_index_requirements() -> GeometryRequirements {
    GeometryRequirements::cooperative(vyre_foundation::CooperativeWidth::Exactly(
        PORTABLE_WORKGROUP_INVOCATIONS,
    ))
}

/// Build a Program that writes `lines[i] = line_number_of(source[i])`.
///
/// Newline bytes belong to the line they terminate, so the generated
/// pipeline scans per-byte "line starts here" increment flags: byte 0 is
/// always line 0, a byte after `\n` starts the next line, and a byte after
/// lone `\r` starts the next line unless the current byte is the `\n` half
/// of `\r\n`.
///
/// This compatibility entry point expects one `DataType::U32` element per
/// source byte and reads the low byte of each word. Use [`line_index_u8`]
/// when the source is packed as one byte per element.
#[must_use]
pub fn line_index(source: &str, lines: &str, n: u32) -> Program {
    line_index_with_block_lanes(source, lines, n, PORTABLE_WORKGROUP_INVOCATIONS)
}

/// Build a line-index Program with explicit lowered block lanes.
#[must_use]
pub fn line_index_with_block_lanes(source: &str, lines: &str, n: u32, block_lanes: u32) -> Program {
    match try_line_index_with_block_lanes(source, lines, n, block_lanes) {
        Ok(program) => program,
        Err(error) => trap_program(LINE_INDEX_OP_ID, Some((lines, DataType::U32)), error),
    }
}

/// Build a line-index Program with lowered launch geometry.
#[must_use]
pub fn line_index_with_geometry(
    source: &str,
    lines: &str,
    n: u32,
    geometry: &vyre_foundation::LaunchGeometry,
) -> Program {
    line_index_with_block_lanes(source, lines, n, geometry.workgroup[0])
}

/// Build a line-index Program over a packed `DataType::U8` source buffer.
///
/// It emits the same per-byte line numbers as [`line_index`] while reducing
/// source input bandwidth from four bytes per logical byte to one.
#[must_use]
pub fn line_index_u8(source: &str, lines: &str, n: u32) -> Program {
    line_index_u8_with_block_lanes(source, lines, n, PORTABLE_WORKGROUP_INVOCATIONS)
}

/// Build a packed `DataType::U8` line-index Program with explicit lowered block lanes.
#[must_use]
pub fn line_index_u8_with_block_lanes(
    source: &str,
    lines: &str,
    n: u32,
    block_lanes: u32,
) -> Program {
    match try_line_index_u8_with_block_lanes(source, lines, n, block_lanes) {
        Ok(program) => program,
        Err(error) => trap_program(LINE_INDEX_OP_ID, Some((lines, DataType::U32)), error),
    }
}

/// Build a packed `DataType::U8` line-index Program with lowered launch geometry.
#[must_use]
pub fn line_index_u8_with_geometry(
    source: &str,
    lines: &str,
    n: u32,
    geometry: &vyre_foundation::LaunchGeometry,
) -> Program {
    line_index_u8_with_block_lanes(source, lines, n, geometry.workgroup[0])
}

fn try_line_index_with_block_lanes(
    source: &str,
    lines: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    try_line_index_with_source_type(source, lines, n, DataType::U32, block_lanes)
}

fn try_line_index_u8_with_block_lanes(
    source: &str,
    lines: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    try_line_index_with_source_type(source, lines, n, DataType::U8, block_lanes)
}

fn try_line_index_with_source_type(
    source: &str,
    lines: &str,
    n: u32,
    source_type: DataType,
    block_lanes: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Ok(empty_line_index_program(source, lines, source_type));
    }
    if !block_lanes.is_power_of_two() || block_lanes < 2 {
        return Err(format!(
            "line_index block_lanes={block_lanes} must be a power of two >= 2. Fix: pass an explicit valid workgroup width."
        ));
    }
    let lanes = block_lanes;
    let flags = format!("__{lines}_line_start_flags");

    let flag_pass = line_start_flags_program(source, &flags, n, source_type, lanes)?;
    let scan_pass = multi_block_prefix_scan_sum_u32_with_block_lanes(&flags, lines, n, lanes);
    if scan_pass.stats().trap() {
        return Err(format!(
            "line_index n={n} could not build its prefix-scan pass. Fix: shard the source before line indexing or repair reduce::multi_block_prefix_scan sizing."
        ));
    }

    vyre_foundation::execution_plan::fusion::fuse_programs(&[flag_pass, scan_pass])
        .map(correct_flag_barrier)
        .map(|program| {
            tag_program(
                LINE_INDEX_OP_ID,
                crate::plumbing::program::outputs::demote_intermediate_outputs(program, lines),
            )
        })
        .map_err(|error| {
            format!(
                "line_index fusion failed for n={n}: {error}. Fix: repair flag/scan fusion instead of falling back to a serial lane-0 loop."
            )
        })
}

/// Correct the fence between the flag pass and scan pass.
///
/// `flag_pass` writes `flags[t] = compute_flag(t)` for invocation `t`.
/// The following scan pass reads `flags[t]` strictly within the same workgroup;
/// no invocation in workgroup `b` ever reads `flags` written by any other
/// workgroup `b'`. `fuse_programs` conservatively marks any arm with an
/// invocation-gated store as requiring a grid-level fence, but source semantics
/// prove that `flags` is block-local. A workgroup barrier (`SeqCst`) orders
/// `flags` writes before the local scan within each workgroup, and any multi-block
/// cross-workgroup synchronization is separately ordered on `block_totals` by
/// `multi_block_prefix_scan`'s own internal `GridSync` barriers.
fn correct_flag_barrier(program: Program) -> Program {
    fn demote_boundary(node: &Node) -> Node {
        match node {
            Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::GridSync,
            } => Node::barrier_with_ordering(vyre_foundation::ir::MemoryOrdering::SeqCst),
            other => other.clone(),
        }
    }

    let entry = program
        .entry()
        .iter()
        .map(|node| match node {
            Node::Region {
                generator,
                source_region,
                body,
            } => Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: Arc::new(body.iter().map(demote_boundary).collect()),
            },
            other => demote_boundary(other),
        })
        .collect();
    program.with_rewritten_entry(entry)
}

fn empty_line_index_program(source: &str, lines: &str, source_type: DataType) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(source, 0, BufferAccess::ReadOnly, source_type).with_count(0),
            BufferDecl::output(lines, 1, DataType::U32)
                .with_count(0)
                .with_output_byte_range(0..0),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(LINE_INDEX_OP_ID, Vec::new())],
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

fn line_start_flags_program(
    source: &str,
    flags: &str,
    n: u32,
    source_type: DataType,
    block_lanes: u32,
) -> Result<Program, String> {
    let t = Expr::InvocationId { axis: 0 };
    let prev_idx = Expr::add(t.clone(), Expr::u32(u32::MAX));
    let output_bytes = output_byte_range(n, "line_index line-start-flags output")?;
    let load_byte = |index: Expr| {
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load(source, index)),
            Expr::u32(0xFF),
        )
    };

    let lane_body = vec![
        Node::let_bind("byte", load_byte(t.clone())),
        Node::let_bind("prev_byte", Expr::u32(0)),
        Node::if_then(
            Expr::lt(Expr::u32(0), t.clone()),
            vec![Node::assign("prev_byte", load_byte(prev_idx))],
        ),
        Node::let_bind("flag", Expr::u32(0)),
        Node::if_then(
            Expr::and(
                Expr::lt(Expr::u32(0), t.clone()),
                Expr::eq(Expr::var("prev_byte"), Expr::u32(0x0A)),
            ),
            vec![Node::assign("flag", Expr::u32(1))],
        ),
        Node::if_then(
            Expr::and(
                Expr::lt(Expr::u32(0), t.clone()),
                Expr::and(
                    Expr::eq(Expr::var("prev_byte"), Expr::u32(0x0D)),
                    Expr::ne(Expr::var("byte"), Expr::u32(0x0A)),
                ),
            ),
            vec![Node::assign("flag", Expr::u32(1))],
        ),
        Node::store(flags, t.clone(), Expr::var("flag")),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(source, 0, BufferAccess::ReadOnly, source_type).with_count(n),
            BufferDecl::storage(flags, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(n)
                .with_pipeline_live_out(true)
                .with_output_byte_range(0..output_bytes),
        ],
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(
            FLAG_OP_ID,
            vec![Node::if_then(Expr::lt(t, Expr::u32(n)), lane_body)],
        )],
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        LINE_INDEX_OP_ID,
        || line_index("source", "lines", 5),
        Some(|| {
            vec![vec![
                vec![0x61, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00],
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
            ]]
        }),
    )
    .with_category("text")
    .with_geometry_requirements(line_index_requirements())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_line_index(input: &[u8]) -> Vec<u32> {
        if input.is_empty() {
            return Vec::new();
        }
        let mut lines = Vec::with_capacity(input.len());
        let mut current_line = 0u32;
        for (i, &b) in input.iter().enumerate() {
            if i > 0 {
                let prev = input[i - 1];
                if prev == b'\n' || (prev == b'\r' && b != b'\n') {
                    current_line += 1;
                }
            }
            lines.push(current_line);
        }
        lines
    }
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
        let program = line_index("source", "lines", PORTABLE_WORKGROUP_INVOCATIONS + 17);
        assert_eq!(
            program.workgroup_size(),
            [PORTABLE_WORKGROUP_INVOCATIONS, 1, 1]
        );
        assert!(program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "__lines_line_start_flags"
                && buffer.is_pipeline_live_out()));
        assert!(!program
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "__lines_line_break_prefix"));
        assert_eq!(
            program
                .buffers()
                .iter()
                .filter(|buffer| buffer.is_output())
                .count(),
            1
        );
    }
    /// WHY: `line_index` composes `flag_pass` and `scan_pass`. `flag_pass` writes
    /// `flags[t] = compute_flag(t)` which is read strictly within the same workgroup by
    /// the subsequent local prefix scan. Because the write is invocation-gated, generic
    /// `fuse_programs` conservatively emitted a `MemoryOrdering::GridSync` whole-grid fence.
    ///
    /// On backends without cooperative launch (such as WGPU), whole-grid fences are cut at
    /// launch boundaries and require a retained read-write graph input to chain state succession.
    /// But `flags` is an intermediate pipeline output, not a retained input, causing WGPU
    /// compilation to fail with `node main holds a whole-grid fence but binds no retained read-write value`.
    ///
    /// This test verifies that for `n <= block_lanes`, `line_index` carries NO `GridSync` barrier,
    /// and for `n > block_lanes`, the top-level flag-to-scan boundary is `SeqCst` while the internal
    /// multi-block prefix scan properly carries its own cross-block `GridSync` boundaries on `block_totals`.
    #[test]
    fn line_index_fence_classification_closes_unwanted_grid_sync_class() {
        for &n in &[1u32, 5, 17, 64, 128, 256] {
            let prog = line_index("source", "lines", n);
            assert!(
                !vyre_foundation::transform::grid_sync_split::contains_grid_sync(&prog),
                "Fix: single-block line_index (n={n} <= 256) must contain no GridSync barriers"
            );

            let prog_u8 = line_index_u8("source", "lines", n);
            assert!(
                !vyre_foundation::transform::grid_sync_split::contains_grid_sync(&prog_u8),
                "Fix: single-block packed-u8 line_index (n={n} <= 256) must contain no GridSync barriers"
            );
        }

        // Multi-block line_index (n > block_lanes) carries GridSync ONLY inside the prefix scan pass.
        for &n in &[512u32, 1024, 2048, 4096] {
            let prog = line_index("source", "lines", n);
            assert!(
                vyre_foundation::transform::grid_sync_split::contains_grid_sync(&prog),
                "Fix: multi-block line_index (n={n} > 256) must contain GridSync for block-total scan"
            );

            // The top-level barrier between flag_pass and scan_pass is SeqCst (not GridSync).
            let is_grid_sync = |node: &Node| {
                matches!(
                    node,
                    Node::Barrier {
                        ordering: vyre_foundation::ir::MemoryOrdering::GridSync
                    }
                )
            };
            let top_level_grid_sync = prog.entry().iter().any(|node| {
                is_grid_sync(node)
                    || vyre_foundation::visit::child_bodies(node)
                        .into_iter()
                        .flatten()
                        .any(is_grid_sync)
            });
            assert!(
                !top_level_grid_sync,
                "Fix: the barrier directly between flag_pass and scan_pass must be SeqCst, not GridSync"
            );
        }
    }

    #[test]
    fn default_builder_uses_portable_workgroup_width() {
        let program = line_index("source", "lines", 17);
        assert!(!program.stats().trap());
        assert_eq!(
            program.workgroup_size(),
            [PORTABLE_WORKGROUP_INVOCATIONS, 1, 1]
        );

        let program_u8 = line_index_u8("source", "lines", 17);
        assert!(!program_u8.stats().trap());
        assert_eq!(
            program_u8.workgroup_size(),
            [PORTABLE_WORKGROUP_INVOCATIONS, 1, 1]
        );
    }

    #[test]
    fn invalid_block_lanes_rejects_and_traps() {
        for invalid_width in [0u32, 1, 3, 5, 7, 100, 300] {
            let program = line_index_with_block_lanes("source", "lines", 17, invalid_width);
            assert!(
                program.stats().trap(),
                "Fix: invalid non-power-of-two or < 2 block_lanes ({invalid_width}) must trap"
            );

            let program_u8 = line_index_u8_with_block_lanes("source", "lines", 17, invalid_width);
            assert!(
                program_u8.stats().trap(),
                "Fix: invalid non-power-of-two or < 2 block_lanes ({invalid_width}) for u8 must trap"
            );
        }
    }

    #[test]
    fn candidate_selection_for_composed_line_index_succeeds_across_all_geometry_widths() {
        for width in [32u32, 64, 128, 256, 512, 1024] {
            let program = line_index_with_block_lanes("source", "lines", 5, width);
            assert!(
                !program.stats().trap(),
                "Fix: line_index must not trap under candidate geometry width {width}"
            );
            assert_eq!(program.workgroup_size(), [width, 1, 1]);

            let program_u8 = line_index_u8_with_block_lanes("source", "lines", 5, width);
            assert!(
                !program_u8.stats().trap(),
                "Fix: line_index_u8 must not trap under candidate geometry width {width}"
            );
            assert_eq!(program_u8.workgroup_size(), [width, 1, 1]);
        }
    }

    #[test]
    fn geometry_lowering_produces_valid_non_trapping_programs() {
        use vyre_foundation::LaunchGeometry;

        for width in [256u32, 512, 1024] {
            let geo = LaunchGeometry {
                workgroup: [width, 1, 1],
                grid: [1, 1, 1],
                elements_per_invocation: 1,
                pipeline_stages: 1,
                shared_bytes: 0,
            };
            let program = line_index_with_geometry("source", "lines", 17, &geo);
            assert!(
                !program.stats().trap(),
                "Fix: line_index_with_geometry must succeed for width {width}"
            );
            assert_eq!(program.workgroup_size(), [width, 1, 1]);
        }
    }
}
