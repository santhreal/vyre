//! Byte histogram primitive over source bytes.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for the 256-bin byte histogram primitive.
pub const BYTE_HISTOGRAM_256_OP_ID: &str = "vyre-primitives::text::byte_histogram_256";

/// Build the reusable histogram body.
#[must_use]
pub fn byte_histogram_256_body(input: &str, histogram: &str, count: u32) -> Vec<Node> {
    let rounds = Expr::div(Expr::add(Expr::u32(count), Expr::u32(255)), Expr::u32(256));
    let load_byte = |index: Expr| {
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load(input, index)),
            Expr::u32(0xFF),
        )
    };

    vec![
        Node::let_bind("lane", Expr::InvocationId { axis: 0 }),
        // Gate the per-lane zero-fill against the bin count. The intended dispatch is
        // one 256-lane workgroup (lane 0..255 = the 256 bins), but whole-workgroup GPU
        // dispatch rounds up, so a >256-lane dispatch would otherwise OOB-write the
        // histogram (memory corruption on CUDA). Transparent for the intended dispatch;
        // caught by the grid-overfire registry gate.
        Node::if_then(
            Expr::lt(Expr::var("lane"), Expr::buf_len(histogram)),
            vec![Node::store(histogram, Expr::var("lane"), Expr::u32(0))],
        ),
        Node::Barrier {
            ordering: vyre_foundation::MemoryOrdering::SeqCst,
        },
        Node::loop_for(
            "round",
            Expr::u32(0),
            rounds,
            vec![
                Node::let_bind(
                    "idx",
                    Expr::add(
                        Expr::mul(Expr::var("round"), Expr::u32(256)),
                        Expr::var("lane"),
                    ),
                ),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(count)),
                    vec![
                        Node::let_bind("byte", load_byte(Expr::var("idx"))),
                        Node::let_bind(
                            "_prev_hist",
                            Expr::atomic_add(histogram, Expr::var("byte"), Expr::u32(1)),
                        ),
                    ],
                ),
            ],
        ),
        Node::Barrier {
            ordering: vyre_foundation::MemoryOrdering::SeqCst,
        },
    ]
}

/// Wrap the histogram body as a child of `parent_op_id`.
#[must_use]
pub fn byte_histogram_256_child(
    parent_op_id: &str,
    input: &str,
    histogram: &str,
    count: u32,
) -> Node {
    byte_histogram_256_child_with_source_type(parent_op_id, input, histogram, count)
}

/// Wrap the packed-`u8` histogram body as a child of `parent_op_id`.
#[must_use]
pub fn byte_histogram_256_u8_child(
    parent_op_id: &str,
    input: &str,
    histogram: &str,
    count: u32,
) -> Node {
    byte_histogram_256_child_with_source_type(parent_op_id, input, histogram, count)
}

fn byte_histogram_256_child_with_source_type(
    parent_op_id: &str,
    input: &str,
    histogram: &str,
    count: u32,
) -> Node {
    Node::Region {
        generator: Ident::from(BYTE_HISTOGRAM_256_OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(byte_histogram_256_body(input, histogram, count)),
    }
}

/// Standalone histogram program for primitive-level conformance.
///
/// This compatibility entry point expects one `DataType::U32` element per
/// source byte and reads the low byte of each word. Use
/// [`byte_histogram_256_u8`] when the source is packed as one byte per
/// element.
#[must_use]
pub fn byte_histogram_256(input: &str, histogram: &str, count: u32) -> Program {
    byte_histogram_256_with_source_type(input, histogram, count, DataType::U32)
}

/// Standalone histogram program over a packed `DataType::U8` source buffer.
///
/// It emits the same 256-bin histogram as [`byte_histogram_256`] while cutting
/// source input bandwidth from four bytes per logical byte to one.
#[must_use]
pub fn byte_histogram_256_u8(input: &str, histogram: &str, count: u32) -> Program {
    byte_histogram_256_with_source_type(input, histogram, count, DataType::U8)
}

fn byte_histogram_256_with_source_type(
    input: &str,
    histogram: &str,
    count: u32,
    source_type: DataType,
) -> Program {
    let input_decl = if source_type == DataType::U8 && count == 0 {
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, source_type)
    } else {
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, source_type).with_count(count.max(1))
    };

    Program::wrapped(
        vec![
            input_decl,
            BufferDecl::output(histogram, 1, DataType::U32)
                .with_count(256)
                .with_output_byte_range(0..256 * 4),
        ],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(BYTE_HISTOGRAM_256_OP_ID),
            source_region: None,
            body: Arc::new(byte_histogram_256_body(input, histogram, count)),
        }],
    )
}

/// Reference oracle for [`byte_histogram_256`].
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_byte_histogram(bytes: &[u8]) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    for &byte in bytes {
        histogram[usize::from(byte)] += 1;
    }
    histogram
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        BYTE_HISTOGRAM_256_OP_ID,
        || byte_histogram_256("bytes", "histogram", 5),
        Some(|| {
            vec![vec![
                crate::wire::pack_bytes_as_u32_slice(&[b'a', b'b', b'a', 0xC3, 0xA9]),
                vec![0; 256 * 4],
            ]]
        }),
        Some(|| {
            let mut histogram = [0u32; 256];
            histogram[usize::from(b'a')] = 2;
            histogram[usize::from(b'b')] = 1;
            histogram[0xC3] = 1;
            histogram[0xA9] = 1;
            vec![vec![crate::wire::pack_u32_slice(&histogram)]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_counts_each_byte() {
        let histogram = reference_byte_histogram(&[b'a', b'b', b'a', 0xC3, 0xA9]);
        assert_eq!(histogram[usize::from(b'a')], 2);
        assert_eq!(histogram[usize::from(b'b')], 1);
        assert_eq!(histogram[0xC3], 1);
        assert_eq!(histogram[0xA9], 1);
    }

    #[test]
    fn packed_u8_program_declares_one_source_byte_per_element() {
        let program = byte_histogram_256_u8("bytes", "histogram", 513);
        let source = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "bytes")
            .expect("Fix: packed-u8 byte histogram source buffer must be declared");
        let histogram = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "histogram")
            .expect("Fix: byte histogram output buffer must be declared");

        assert_eq!(source.element(), DataType::U8);
        assert_eq!(source.count(), 513);
        assert_eq!(histogram.element(), DataType::U32);
        assert_eq!(histogram.count(), 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
    }

    #[test]
    fn masks_high_bit_source_and_records_no_interpreter_oob() {
        use vyre_reference::value::Value;
        // A U32 source element > 255 must fold into the 256-bin histogram via the
        // `& 0xFF` bin mask, never atomic-add past the last bin. Assert ZERO
        // interpreter OOB accesses (the mask keeps the atomic in bounds, not the
        // interpreter's silent drop) and that the low byte's bin is incremented.
        // Removing the mask would OOB the atomic scatter (memory corruption on CUDA)
        // and this test would see report.total() > 0.
        let program = byte_histogram_256("bytes", "histogram", 1);
        let (outputs, report) = vyre_reference::reference_eval_oob_report(
            &program,
            &[
                Value::from(crate::wire::pack_u32_slice(&[0x0141])), // > 255, low byte 0x41
                Value::from(vec![0u8; 256 * 4]),
            ],
        )
        .expect("Fix: byte_histogram_256 must reference-evaluate a high-bit source element");
        assert_eq!(
            report.total(),
            0,
            "Fix: masked bin index must stay in bounds without relying on interpreter OOB masking"
        );
        let histogram = crate::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        assert_eq!(
            histogram[0x41], 1,
            "Fix: 0x0141 must count into bin 0x41 via the `& 0xFF` mask"
        );
        assert_eq!(
            histogram.iter().sum::<u32>(),
            1,
            "exactly one byte counted, no stray OOB write elsewhere"
        );
    }
}
