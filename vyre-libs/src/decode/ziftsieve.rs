//! LZ4 sequence-index literal-copy: one module, one op id.
//!
//! LZ4-style formats have serial sequence discovery but parallel literal
//! copying once an index exists. This module is the reusable second stage:
//! one lane per sequence copies `[literal_start, literal_start + literal_len)`
//! into the prefix-summed output offset. Producers may be a host oracle, any
//! accelerator backend, or a future persistent decode megakernel as long as
//! they satisfy the same sequence-index contract.
//!
//! The IR builder, the host oracle, the fixture registration and the
//! family-scoped entry point all live here. The op id is
//! `vyre-libs::decode::ziftsieve_literal_copy`.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decode_output_buffer};

const FAMILY_PREFIX: &str = "decode_ziftsieve";

/// Canonical primitive op id.
pub const OP_ID: &str = "vyre-libs::decode::ziftsieve_literal_copy";
/// One invocation processes one indexed LZ4 sequence.
pub const WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];
/// Defensive upper bound for one compressed block.
pub const MAX_BLOCK_SIZE: usize = 4 * 1024 * 1024;
/// Defensive upper bound for sequence count in one block.
pub const MAX_SEQUENCES_PER_BLOCK: usize = 100_000;

/// Buffer names one indexed literal-copy dispatch binds.
///
/// Every field is a `&str`, so a positional list of five names is a positional
/// list wearing field names: transposing `seq_literal_len` with
/// `seq_literal_offset` compiles, reads as deliberate from either side, and
/// copies each literal run to the length it should have had rather than to its
/// output offset. There is no constructor, so a struct literal names every
/// binding at the construction site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZiftsieveBuffers<'a> {
    /// Compressed block words, one byte per `u32`.
    pub input: &'a str,
    /// Decoded literal words, `max_output` elements.
    pub output: &'a str,
    /// Per-sequence offset of the literal run inside `input`.
    pub seq_literal_start: &'a str,
    /// Per-sequence literal run length.
    pub seq_literal_len: &'a str,
    /// Per-sequence prefix-summed offset inside `output`.
    pub seq_literal_offset: &'a str,
}

impl ZiftsieveBuffers<'static> {
    /// The canonical binding names for a literal-copy program.
    ///
    /// A caller with no naming of its own gets one here instead of inventing
    /// five strings, and every program built from it declares the same bindings
    /// in the same order, which is what lets two such programs be compared.
    pub const CANONICAL: Self = Self {
        input: "input",
        output: "output",
        seq_literal_start: "seq_start",
        seq_literal_len: "seq_len",
        seq_literal_offset: "seq_off",
    };
}

/// Extents that size the buffers a literal-copy program declares.
///
/// `input_len` of zero leaves the input count unbounded, which is how a caller
/// that does not know the block size declares it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ZiftsieveExtents {
    /// Element count of `input`, or zero to leave it undeclared.
    pub input_len: u32,
    /// Number of indexed sequences, one lane each.
    pub seq_count: u32,
    /// Element count of `output`, the cap every store is gated on.
    pub max_output: u32,
}

/// Build the primitive body for indexed literal copy.
#[must_use]
pub fn ziftsieve_literal_copy_body(buffers: ZiftsieveBuffers<'_>, seq_count: u32) -> Vec<Node> {
    let ZiftsieveBuffers {
        input,
        output,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
    } = buffers;
    vec![
        Node::let_bind("seq_idx", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("seq_idx"), Expr::u32(seq_count)),
            vec![
                Node::let_bind(
                    "literal_start",
                    Expr::load(seq_literal_start, Expr::var("seq_idx")),
                ),
                Node::let_bind(
                    "literal_len",
                    Expr::load(seq_literal_len, Expr::var("seq_idx")),
                ),
                Node::let_bind(
                    "literal_offset",
                    Expr::load(seq_literal_offset, Expr::var("seq_idx")),
                ),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::var("literal_len"),
                    // Gate the data-derived copy on BOTH buffer bounds with control flow
                    // (an `if_then`, NOT `Expr::select`: select still evaluates the OOB
                    // load on a real GPU). The seq_* indices are unvalidated producer
                    // input, so an out-of-contract `literal_start`/`literal_offset` would
                    // otherwise be a raw OOB read and OOB write, which is undefined
                    // behaviour on a real GPU. This puts the documented "drops stores whose
                    // `literal_offset + i >= max_output`" cap INTO the IR instead of
                    // relying on unreliable driver OOB behavior (see vyre-reference
                    // oob.rs: "some clamp, some return zero, some crash"). Transparent to
                    // every valid input (the producer contract keeps both indices in
                    // bounds) and byte-identical to the interpreter's existing silent
                    // OOB-store drop on a zero-initialized output.
                    vec![Node::if_then(
                        Expr::and(
                            Expr::lt(
                                Expr::add(Expr::var("literal_start"), Expr::var("i")),
                                Expr::buf_len(input),
                            ),
                            Expr::lt(
                                Expr::add(Expr::var("literal_offset"), Expr::var("i")),
                                Expr::buf_len(output),
                            ),
                        ),
                        vec![
                            Node::let_bind(
                                "src",
                                Expr::load(
                                    input,
                                    Expr::add(Expr::var("literal_start"), Expr::var("i")),
                                ),
                            ),
                            Node::store(
                                output,
                                Expr::add(Expr::var("literal_offset"), Expr::var("i")),
                                Expr::var("src"),
                            ),
                        ],
                    )],
                ),
            ],
        ),
    ]
}

/// Build a Program that copies indexed LZ4 literals in parallel over the
/// buffer names the caller supplies.
#[must_use]
pub fn ziftsieve_literal_copy(buffers: ZiftsieveBuffers<'_>, extents: ZiftsieveExtents) -> Program {
    let ZiftsieveBuffers {
        input,
        output,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
    } = buffers;
    let ZiftsieveExtents {
        input_len,
        seq_count,
        max_output,
    } = extents;
    let body = ziftsieve_literal_copy_body(buffers, seq_count);

    let input_decl = BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32);
    let input_decl = if input_len == 0 {
        input_decl
    } else {
        input_decl.with_count(input_len)
    };

    Program::wrapped(
        vec![
            input_decl,
            BufferDecl::storage(seq_literal_start, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(seq_count.max(1)),
            BufferDecl::storage(seq_literal_len, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(seq_count.max(1)),
            BufferDecl::storage(seq_literal_offset, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(seq_count.max(1)),
            BufferDecl::storage(output, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(max_output.max(1)),
        ],
        WORKGROUP_SIZE,
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Build a literal-copy Program whose generic `input` and `output` bindings are
/// rewritten to family-scoped names.
///
/// A fused decode kernel binds several decoders in one program, where the
/// generic names collide. An explicitly named binding is left as the caller
/// wrote it.
#[must_use]
pub fn ziftsieve_gpu(buffers: ZiftsieveBuffers<'_>, extents: ZiftsieveExtents) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, buffers.input);
    let output = scoped_decode_output_buffer(
        FAMILY_PREFIX,
        "output",
        buffers.output,
        &["output", "decoded"],
    );
    ziftsieve_literal_copy(
        ZiftsieveBuffers {
            input: &input,
            output: &output,
            ..buffers
        },
        extents,
    )
}

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let input =
        vyre_primitives::wire::pack_u32_slice(&[0x10, b'A' as u32, 0x20, b'B' as u32, b'C' as u32]);
    let seq_literal_start = vyre_primitives::wire::pack_u32_slice(&[1, 3]);
    let seq_literal_len = vyre_primitives::wire::pack_u32_slice(&[1, 2]);
    let seq_literal_offset = vyre_primitives::wire::pack_u32_slice(&[0, 1]);
    vec![vec![
        input,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
        vec![0u8; 3 * 4],
    ]]
}

const EXPECTED_ZIFTSIEVE_LITERAL_BYTES: [u8; 12] = [65, 0, 0, 0, 66, 0, 0, 0, 67, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            ziftsieve_literal_copy(
                ZiftsieveBuffers::CANONICAL,
                ZiftsieveExtents {
                    input_len: 5,
                    seq_count: 2,
                    max_output: 3,
                },
            )
        },
        Some(fixture_inputs),
        Some(|| vec![vec![EXPECTED_ZIFTSIEVE_LITERAL_BYTES.to_vec()]]),
    )
}
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod primitive_tests {
    use super::*;
    use crate::fixture_bytes::bytes_to_u32;
    use crate::fixture_bytes::eval_bytes;
    use vyre_reference::composition_witness::ziftsieve_extract_literals_witness as ziftsieve_reference_extract_literals;

    fn literals(
        input: &[u8],
        seq_starts: &[u32],
        seq_lens: &[u32],
        seq_offsets: &[u32],
    ) -> Vec<u32> {
        let seq_count = seq_starts.len() as u32;
        let max_output = seq_lens.iter().copied().sum::<u32>();
        let input_words = input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>();
        let program = ziftsieve_literal_copy(
            ZiftsieveBuffers::CANONICAL,
            ZiftsieveExtents {
                input_len: input.len() as u32,
                seq_count,
                max_output,
            },
        );
        let outputs = eval_bytes(
            "ziftsieve_literal_copy",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&input_words),
                vyre_primitives::wire::pack_u32_slice(seq_starts),
                vyre_primitives::wire::pack_u32_slice(seq_lens),
                vyre_primitives::wire::pack_u32_slice(seq_offsets),
                vec![0u8; (max_output.max(1) as usize) * 4],
            ],
        );
        bytes_to_u32(&outputs[0])
            .into_iter()
            .take(max_output as usize)
            .collect()
    }

    #[test]
    fn single_literal() {
        assert_eq!(literals(&[0x10, b'A'], &[1], &[1], &[0]), vec![b'A' as u32]);
    }

    #[test]
    fn two_sequences() {
        assert_eq!(
            literals(&[0x10, b'A', 0x20, b'B', b'C'], &[1, 3], &[1, 2], &[0, 1]),
            vec![b'A' as u32, b'B' as u32, b'C' as u32]
        );
    }

    #[test]
    fn zero_literal_sequence_is_nop() {
        assert_eq!(
            literals(&[0x00, 0x10, b'A'], &[0], &[0], &[0]),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn reference_extracts_simple_literal() {
        let result = ziftsieve_reference_extract_literals(&[0x10, b'A'], 1024).unwrap();
        assert_eq!(result.literals, b"A");
        assert_eq!(result.decoded_len, 1);
        assert!(!result.truncated());
    }

    #[test]
    fn reference_extracts_with_match_skip() {
        let data = [0x11, b'A', 0x01, 0x00];
        let result = ziftsieve_reference_extract_literals(&data, 1024).unwrap();
        assert_eq!(result.literals, b"A");
        assert!(!result.truncated());
    }

    #[test]
    fn reference_rejects_truncated_literal() {
        let err = ziftsieve_reference_extract_literals(&[0x20, b'A'], 1024).unwrap_err();
        assert!(err.contains("truncated") || err.contains("literal"));
    }

    #[test]
    fn reference_accepts_exact_max_sequence_count() {
        let mut data = Vec::new();
        for _ in 1..MAX_SEQUENCES_PER_BLOCK {
            data.push(0x10);
            data.push(b'X');
            data.extend_from_slice(&[0x00, 0x00]);
        }
        data.push(0x10);
        data.push(b'X');

        let result = ziftsieve_reference_extract_literals(&data, MAX_SEQUENCES_PER_BLOCK)
            .expect("Fix: MAX_SEQUENCES_PER_BLOCK is an inclusive maximum, not an exclusive one.");
        assert_eq!(result.literals.len(), MAX_SEQUENCES_PER_BLOCK);
        assert!(result.literals.iter().all(|&byte| byte == b'X'));
        assert!(!result.truncated());
    }

    #[test]
    fn reference_rejects_too_many_sequences() {
        let mut data = Vec::new();
        for _ in 0..=MAX_SEQUENCES_PER_BLOCK {
            data.push(0x10);
            data.push(b'X');
            data.extend_from_slice(&[0x00, 0x00]);
        }
        let err = ziftsieve_reference_extract_literals(&data, 1024).unwrap_err();
        assert!(err.contains("sequence") || err.contains("MAX"));
    }

    /// Run the copy program with a caller-controlled output cap and a
    /// SENTINEL-prefilled output buffer, returning the raw output words so a test
    /// can prove which slots the gate left untouched. Unlike `run`, this does not
    /// zero-init or truncate (it exposes the exact OOB behavior).
    fn run_with_sentinel(
        input_words: &[u32],
        seq_starts: &[u32],
        seq_lens: &[u32],
        seq_offsets: &[u32],
        max_output: u32,
        sentinel: u32,
    ) -> Vec<u32> {
        let program = ziftsieve_literal_copy(
            ZiftsieveBuffers::CANONICAL,
            ZiftsieveExtents {
                input_len: input_words.len() as u32,
                seq_count: seq_starts.len() as u32,
                max_output,
            },
        );
        let inputs = vec![
            vyre_primitives::wire::pack_u32_slice(input_words),
            vyre_primitives::wire::pack_u32_slice(seq_starts),
            vyre_primitives::wire::pack_u32_slice(seq_lens),
            vyre_primitives::wire::pack_u32_slice(seq_offsets),
            vyre_primitives::wire::pack_u32_slice(&vec![sentinel; max_output.max(1) as usize]),
        ];
        let outputs = eval_bytes("ziftsieve", &program, inputs);
        vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0])
    }

    #[test]
    fn out_of_contract_offset_drops_stores_past_output_cap() {
        // The seq_* indices are UNVALIDATED producer input. A literal whose copy
        // runs past the `max_output` cap must have its out-of-range stores dropped
        // BY THE IR gate (the documented contract), not by unreliable driver OOB
        // behavior. Proven by a non-zero sentinel that survives past the cap and by
        // the run not faulting. Buffer holds 3 slots; the sequence starts at offset
        // 1 with length 4, so slots 3 and 4 are past the cap and must be dropped.
        const SENTINEL: u32 = 0xDEAD_BEEF;
        let words = run_with_sentinel(
            &[b'A' as u32, b'B' as u32, b'C' as u32, b'D' as u32],
            &[0],
            &[4],
            &[1],
            3,
            SENTINEL,
        );
        // slot 0: never written (offset starts at 1) → sentinel preserved.
        // slots 1,2: in-bounds copies of input[0]=A, input[1]=B.
        // slots 3,4: past the 3-slot cap → dropped (no panic, no corruption).
        assert_eq!(
            words,
            vec![SENTINEL, b'A' as u32, b'B' as u32],
            "Fix: stores past the output cap must be dropped by the IR gate, untouched slots keep their prior value"
        );
    }

    #[test]
    fn out_of_contract_literal_start_gates_oob_source_reads() {
        // A `literal_start`/`literal_len` that runs past the input buffer must have
        // its out-of-range SOURCE READS gated away entirely (no OOB load, which is
        // undefined behaviour on a real GPU), leaving the corresponding output slots untouched. This distinguishes
        // the control-flow gate from the OLD ungated IR: the old code zero-fill-loaded
        // the OOB source and stored 0 (→ [B, 0, 0, SENTINEL]); the gate skips the whole
        // iteration (→ [B, SENTINEL, SENTINEL, SENTINEL]).
        const SENTINEL: u32 = 0x1234_5678;
        let words = run_with_sentinel(
            &[b'A' as u32, b'B' as u32], // input_len = 2
            &[1],                        // start at the last valid index
            &[3],                        // reads input[1] (ok), input[2],input[3] (OOB)
            &[0],
            4,
            SENTINEL,
        );
        // i=0: input[1]=B → output[0]=B (both in bounds).
        // i=1: input[2] OOB → iteration skipped → output[1] keeps sentinel.
        // i=2: input[3] OOB → skipped → output[2] keeps sentinel.
        // output[3]: never touched → sentinel.
        assert_eq!(
            words,
            vec![b'B' as u32, SENTINEL, SENTINEL, SENTINEL],
            "Fix: OOB source reads must be skipped by the IR gate (no OOB load), leaving output untouched"
        );
    }

    #[test]
    fn out_of_contract_copy_records_zero_interpreter_oob_accesses() {
        // The whole point of the gate: on hostile input the program must NOT rely on
        // the interpreter's silent OOB masking (zero-fill loads / dropped stores). A
        // correctly-gated copy skips the out-of-range access with control flow, so
        // reference_eval reports ZERO OOB accesses even though the sequence overshoots
        // the 3-slot output. The pre-fix ungated store would OOB-write slots 3,4 past
        // the buffer → nonzero, which is what a real GPU would corrupt.
        let program = ziftsieve_literal_copy(
            ZiftsieveBuffers::CANONICAL,
            ZiftsieveExtents {
                input_len: 4,
                seq_count: 1,
                max_output: 3,
            },
        );
        let (_outputs, report) = vyre_reference::reference_eval_oob_report(
            &program,
            &[
                vyre_reference::value::Value::from(vyre_primitives::wire::pack_u32_slice(&[
                    10, 20, 30, 40,
                ])),
                vyre_reference::value::Value::from(vyre_primitives::wire::pack_u32_slice(&[0])),
                vyre_reference::value::Value::from(
                    // literal_start
                    vyre_primitives::wire::pack_u32_slice(&[4]),
                ),
                vyre_reference::value::Value::from(
                    // literal_len overshoots the cap
                    vyre_primitives::wire::pack_u32_slice(&[1]),
                ),
                vyre_reference::value::Value::from(
                    // literal_offset → slots 1..4
                    vyre_primitives::wire::pack_u32_slice(&[0u32; 3]),
                ),
            ],
        )
        .expect("Fix: ziftsieve copy must reference-evaluate");
        assert_eq!(
            report.total(),
            0,
            "Fix: the bounds-gated copy must never trigger interpreter OOB masking on hostile input"
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// WHY: the generic `input` and `output` names are what a fused decode
    /// kernel collides on, so the rewrite in `ziftsieve_gpu` is the only thing
    /// that separates it from the explicitly named builder. An explicit caller
    /// name must survive the rewrite, or composition by name breaks.
    #[test]
    fn generic_binding_names_are_family_scoped_and_explicit_ones_are_kept() {
        let scoped = ziftsieve_gpu(ZiftsieveBuffers::CANONICAL, ZiftsieveExtents::default());
        let names: Vec<&str> = scoped
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect();
        assert!(
            names.contains(&"__vyre_decode_ziftsieve_input")
                && names.contains(&"__vyre_decode_ziftsieve_output"),
            "Fix: generic `input`/`output` must be rewritten to family-scoped names, got {names:?}"
        );
        let explicit = ziftsieve_gpu(
            ZiftsieveBuffers {
                input: "block_words",
                output: "literals",
                ..ZiftsieveBuffers::CANONICAL
            },
            ZiftsieveExtents::default(),
        );
        let names: Vec<&str> = explicit
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect();
        assert!(
            names.contains(&"block_words") && names.contains(&"literals"),
            "Fix: an explicit caller name must be preserved, got {names:?}"
        );
    }

    /// WHY: the module carries exactly one registered op id after the two
    /// pre-move copies collapsed. A second id reaching the program identity is
    /// the duplication this collapse removed, so every builder here must name
    /// the same single operation.
    #[test]
    fn one_op_id_reaches_every_builder() {
        use vyre_foundation::ir::Node;
        use vyre_foundation::visit::walk_nodes;

        let scoped = ziftsieve_gpu(ZiftsieveBuffers::CANONICAL, ZiftsieveExtents::default());
        let explicit =
            ziftsieve_literal_copy(ZiftsieveBuffers::CANONICAL, ZiftsieveExtents::default());
        for program in [&scoped, &explicit] {
            let mut ids: Vec<String> = Vec::new();
            walk_nodes(program, |node| {
                if let Node::Region { generator, .. } = node {
                    ids.push(generator.as_str().to_string());
                }
            });
            assert_eq!(
                ids,
                vec![OP_ID.to_string()],
                "Fix: every ziftsieve builder must name only `{OP_ID}`, got {ids:?}"
            );
        }
    }
}
