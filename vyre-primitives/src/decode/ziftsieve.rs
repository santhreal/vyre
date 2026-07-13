//! LZ4 sequence-index literal-copy primitive.
//!
//! LZ4-style formats have serial sequence discovery but parallel literal
//! copying once an index exists. This primitive is the reusable second stage:
//! one lane per sequence copies `[literal_start, literal_start + literal_len)`
//! into the prefix-summed output offset. Producers may be CPU, CUDA, WGPU, or
//! a future persistent decode megakernel as long as they satisfy the same
//! sequence-index contract.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical primitive op id.
pub const OP_ID: &str = "vyre-primitives::decode::ziftsieve_literal_copy";
/// One invocation processes one indexed LZ4 sequence.
pub const WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];
/// Defensive upper bound for one compressed block.
pub const MAX_BLOCK_SIZE: usize = 4 * 1024 * 1024;
/// Defensive upper bound for sequence count in one block.
pub const MAX_SEQUENCES_PER_BLOCK: usize = 100_000;

/// Result of a reference LZ4 literal extraction.
///
/// `literals` holds the decoded bytes, CAPPED at the caller's `max_output`: the
/// same fixed-output-buffer bound the GPU `ziftsieve_literal_copy` kernel enforces
/// (it drops stores whose `literal_offset + i >= max_output`). `decoded_len` is the
/// TRUE uncapped output length, the sum of every sequence's literal length, so a
/// caller can detect a capped decode via [`ZiftsieveExtract::truncated`]. A bare
/// `Vec<u8>` could not distinguish a complete decode from one silently truncated at
/// the cap (a silent recall-loss gap, Law 10); the GPU path already exposes this
/// host-side because the consumer builds the prefix-summed offsets and thus knows
/// `offsets[last] + lens[last]` vs `max_output`. This mirrors [`super::inflate::CpuInflateResult`].
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZiftsieveExtract {
    /// Decoded literal bytes, capped at the caller's `max_output`.
    pub literals: Vec<u8>,
    /// True total decoded length across all sequences, BEFORE the `max_output` cap.
    pub decoded_len: usize,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl ZiftsieveExtract {
    /// True iff the decode was capped at `max_output` (bytes past the cap were
    /// dropped). When true, `literals.len() == max_output < decoded_len`.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.decoded_len > self.literals.len()
    }
}

/// Host-side reference: sequential LZ4 literal extraction.
///
/// Returns the decoded bytes (capped at `max_output`) plus the true uncapped
/// [`ZiftsieveExtract::decoded_len`], so a capped decode is observable rather than a
/// silent recall loss (Law 10). Every malformed-input path still fails LOUD.
///
/// # Errors
///
/// Returns an actionable error string on malformed input. Every error message
/// includes a `Fix:` tag.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn ziftsieve_reference_extract_literals(
    compressed: &[u8],
    max_output: usize,
) -> Result<ZiftsieveExtract, String> {
    let initial_cap = compressed
        .len()
        .saturating_mul(2)
        .min(max_output)
        .min(MAX_BLOCK_SIZE);
    let mut literals = Vec::with_capacity(initial_cap);
    let mut decoded_len = 0usize;
    let mut pos = 0usize;
    let mut sequence_count = 0usize;

    while pos < compressed.len() {
        sequence_count += 1;
        if sequence_count > MAX_SEQUENCES_PER_BLOCK {
            return Err(format!(
                "too many LZ4 sequences (max {MAX_SEQUENCES_PER_BLOCK}). \
                 Fix: use a smaller LZ4 block or increase MAX_SEQUENCES_PER_BLOCK"
            ));
        }

        let token = compressed[pos];
        pos += 1;

        let literal_len = (token >> 4) as usize;
        let match_len = (token & 0x0F) as usize;

        let literal_len = if literal_len == 15 {
            decode_length(compressed, &mut pos, literal_len)?
        } else {
            literal_len
        };

        if literal_len > MAX_BLOCK_SIZE {
            return Err(format!(
                "literal length {literal_len} exceeds MAX_BLOCK_SIZE {MAX_BLOCK_SIZE}. \
                 Fix: use a valid LZ4 stream"
            ));
        }

        if pos + literal_len > compressed.len() {
            return Err(format!(
                "literal exceeds block bounds at offset {pos}. \
                 Fix: use a valid LZ4 stream"
            ));
        }

        // Count every valid in-stream literal toward the TRUE decoded length before
        // applying the `max_output` cap, so the caller can detect a capped decode.
        decoded_len = decoded_len.saturating_add(literal_len);
        let remaining_output = max_output.saturating_sub(literals.len());
        let to_copy = literal_len.min(remaining_output);
        if to_copy > 0 {
            literals.extend_from_slice(&compressed[pos..pos + to_copy]);
        }
        pos += literal_len;

        if pos < compressed.len() {
            if pos + 2 > compressed.len() {
                return Err(format!(
                    "truncated match offset at offset {pos}. \
                     Fix: use a complete LZ4 stream"
                ));
            }
            pos += 2;

            if match_len == 15 {
                let _match_len_extension = decode_length(compressed, &mut pos, match_len)?;
            }
        }
    }

    Ok(ZiftsieveExtract {
        literals,
        decoded_len,
    })
}

fn decode_length(data: &[u8], pos: &mut usize, initial: usize) -> Result<usize, String> {
    let mut len = initial;
    loop {
        if *pos >= data.len() {
            return Err(format!(
                "truncated length encoding at offset {pos}. \
                 Fix: use a complete LZ4 stream"
            ));
        }
        let byte = data[*pos];
        *pos += 1;
        len = len.checked_add(byte as usize).ok_or_else(|| {
            "length overflow in variable-length encoding. Fix: use a valid LZ4 stream".to_string()
        })?;
        if byte < 255 {
            break;
        }
        if len > MAX_BLOCK_SIZE {
            return Err(format!(
                "length {len} exceeds MAX_BLOCK_SIZE {MAX_BLOCK_SIZE}. \
                 Fix: use a valid LZ4 stream"
            ));
        }
    }
    Ok(len)
}

/// Build the primitive body for indexed literal copy.
#[must_use]
pub fn ziftsieve_literal_copy_body(
    input: &str,
    output: &str,
    seq_literal_start: &str,
    seq_literal_len: &str,
    seq_literal_offset: &str,
    seq_count: u32,
) -> Vec<Node> {
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
                    // otherwise be a raw OOB read (UB on CUDA) and OOB write (memory
                    // corruption on CUDA). This puts the documented "drops stores whose
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

/// Build a Program that copies indexed LZ4 literals in parallel.
#[must_use]
pub fn ziftsieve_literal_copy(
    input: &str,
    output: &str,
    seq_literal_start: &str,
    seq_literal_len: &str,
    seq_literal_offset: &str,
    input_len: u32,
    seq_count: u32,
    max_output: u32,
) -> Program {
    ziftsieve_literal_copy_with_op_id(
        OP_ID,
        input,
        output,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
        input_len,
        seq_count,
        max_output,
    )
}

/// Build a Program with a caller-provided op id.
///
/// Composition crates use this to preserve their public inventory id while
/// reusing the primitive-owned IR builder.
#[must_use]
pub fn ziftsieve_literal_copy_with_op_id(
    op_id: &str,
    input: &str,
    output: &str,
    seq_literal_start: &str,
    seq_literal_len: &str,
    seq_literal_offset: &str,
    input_len: u32,
    seq_count: u32,
    max_output: u32,
) -> Program {
    let body = ziftsieve_literal_copy_body(
        input,
        output,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
        seq_count,
    );

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
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

#[cfg(feature = "inventory-registry")]
fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let input = crate::wire::pack_u32_slice(&[0x10, b'A' as u32, 0x20, b'B' as u32, b'C' as u32]);
    let seq_literal_start = crate::wire::pack_u32_slice(&[1, 3]);
    let seq_literal_len = crate::wire::pack_u32_slice(&[1, 2]);
    let seq_literal_offset = crate::wire::pack_u32_slice(&[0, 1]);
    vec![vec![
        input,
        seq_literal_start,
        seq_literal_len,
        seq_literal_offset,
        vec![0u8; 3 * 4],
    ]]
}

#[cfg(feature = "inventory-registry")]
fn fixture_outputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![crate::wire::pack_u32_slice(&[
        b'A' as u32,
        b'B' as u32,
        b'C' as u32,
    ])]]
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || ziftsieve_literal_copy("input", "output", "seq_start", "seq_len", "seq_off", 5, 2, 3),
        Some(fixture_inputs),
        Some(fixture_outputs),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn run(input: &[u8], seq_starts: &[u32], seq_lens: &[u32], seq_offsets: &[u32]) -> Vec<u32> {
        let seq_count = seq_starts.len() as u32;
        let max_output = seq_lens.iter().copied().sum::<u32>();
        let input_words = input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>();
        let program = ziftsieve_literal_copy(
            "input",
            "output",
            "seq_start",
            "seq_len",
            "seq_off",
            input.len() as u32,
            seq_count,
            max_output,
        );
        let inputs = vec![
            Value::from(crate::wire::pack_u32_slice(&input_words)),
            Value::from(crate::wire::pack_u32_slice(seq_starts)),
            Value::from(crate::wire::pack_u32_slice(seq_lens)),
            Value::from(crate::wire::pack_u32_slice(seq_offsets)),
            Value::from(vec![0u8; (max_output.max(1) as usize) * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: ziftsieve literal-copy primitive must run.");
        let words = crate::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        words.into_iter().take(max_output as usize).collect()
    }

    #[test]
    fn single_literal() {
        assert_eq!(run(&[0x10, b'A'], &[1], &[1], &[0]), vec![b'A' as u32]);
    }

    #[test]
    fn two_sequences() {
        assert_eq!(
            run(&[0x10, b'A', 0x20, b'B', b'C'], &[1, 3], &[1, 2], &[0, 1]),
            vec![b'A' as u32, b'B' as u32, b'C' as u32]
        );
    }

    #[test]
    fn zero_literal_sequence_is_nop() {
        assert_eq!(
            run(&[0x00, 0x10, b'A'], &[0], &[0], &[0]),
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
            "input",
            "output",
            "seq_start",
            "seq_len",
            "seq_off",
            input_words.len() as u32,
            seq_starts.len() as u32,
            max_output,
        );
        let inputs = vec![
            Value::from(crate::wire::pack_u32_slice(input_words)),
            Value::from(crate::wire::pack_u32_slice(seq_starts)),
            Value::from(crate::wire::pack_u32_slice(seq_lens)),
            Value::from(crate::wire::pack_u32_slice(seq_offsets)),
            Value::from(crate::wire::pack_u32_slice(&vec![
                sentinel;
                max_output.max(1) as usize
            ])),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: out-of-contract ziftsieve copy must not fault the interpreter");
        crate::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
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
        // its out-of-range SOURCE READS gated away entirely (no OOB load. UB on
        // CUDA), leaving the corresponding output slots untouched. This distinguishes
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
            "input",
            "output",
            "seq_start",
            "seq_len",
            "seq_off",
            4,
            1,
            3,
        );
        let (_outputs, report) = vyre_reference::reference_eval_oob_report(
            &program,
            &[
                Value::from(crate::wire::pack_u32_slice(&[10, 20, 30, 40])),
                Value::from(crate::wire::pack_u32_slice(&[0])), // literal_start
                Value::from(crate::wire::pack_u32_slice(&[4])), // literal_len overshoots the cap
                Value::from(crate::wire::pack_u32_slice(&[1])), // literal_offset → slots 1..4
                Value::from(crate::wire::pack_u32_slice(&[0u32; 3])),
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
