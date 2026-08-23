//! GPU-accelerated byte histogram + encoding classification.
//!
//! Replaces the CPU-only sliding-window stats used by Mozilla-style
//! universalchardet with a single-dispatch GPU histogram kernel. Each
//! work-item owns one byte value (0..255) and counts occurrences across
//! the entire input. Thread 0 then reads the 256-bin histogram and
//! applies a compact heuristic classifier.
//!
//! # Design notes
//!
//! - Single workgroup `256,1,1` keeps the classification exact (no
//!   cross-workgroup reduction needed). The histogram is the bottleneck
//!   for multi-MB scans; a single SM can saturate most of device memory
//!   bandwidth with perfectly coalesced strided loads.
//! - N-gram frequencies are **not** computed on-GPU yet; the task
//!   explicitly permits leaving the small-N classifier on CPU and
//!   focusing on the byte-histogram pass (PHASE2_DECODE MEDIUM).
//! - The host reference mirrors the GPU heuristics so conformance can
//!   prove the on-device path without routing production work through it.

use crate::text::byte_histogram_256_child;
use crate::text::encoding_classify_child;
#[cfg(test)]
use crate::text::{ENC_ASCII, ENC_ISO8859_1, ENC_UTF16LE, ENC_UTF8};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

#[cfg(test)]
use crate::buffer_names::fixed_name;
use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decode_output_buffer};

const OP_ID: &str = "vyre-libs::decode::encodex";
const FAMILY_PREFIX: &str = "decode_encodex";
const HISTOGRAM_BUFFER: &str = "__vyre_decode_encodex_histogram";

// Cross-domain reuse: same LE u32-pack byte layout as the matching
// dialect's storage-buffer inputs. Single source of truth in
// `scan::dispatch_io::pack_u32_slice` - was a third inline copy here.
use vyre_primitives::wire::pack_u32_slice as pack_words;

/// Build a Program that computes a 256-bin byte histogram over `input`
/// and writes the detected encoding-id to `output`.
///
/// The input buffer carries one byte per `u32` element (same convention
/// used by `vyre-libs::decode::base64` and `hex`).  The histogram is
/// exposed as a `read_write` buffer so callers can read it back for
/// their own CPU-side refinement if desired.
///
/// ```ignore
/// use vyre_libs::decode::encodex::encodex_gpu;
///
/// let program = encodex_gpu("bytes", "encoding", 1024);
/// assert_eq!(program.buffers().len(), 3);
/// ```
#[must_use]
pub fn encodex_gpu(input: &str, output: &str, count: u32) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let output = scoped_decode_output_buffer(
        FAMILY_PREFIX,
        "encoding_id",
        output,
        &["encoding_id", "output"],
    );
    let histogram = HISTOGRAM_BUFFER.to_string();
    let body = vec![
        byte_histogram_256_child(OP_ID, &input, &histogram, count),
        encoding_classify_child(OP_ID, &histogram, &output, count),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(&input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count.max(1)),
            BufferDecl::read_write(&histogram, 1, DataType::U32).with_count(256),
            BufferDecl::output(&output, 2, DataType::U32).with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

// ---------------------------------------------------------------------------
// Fixtures & harness
// ---------------------------------------------------------------------------

/// Deterministic fixture inputs for encodex operation registration.
const FIXTURE_INPUTS: &[&[u8]] = &[
    b"Hello",
    &[0xC3, 0xA9, 0xC3, 0xA9, b'!'],
    &[0x00, 0x00, 0x00, 0x41, 0x42],
    &[0xE9, 0xE8, 0xEA, 0xEB, 0xEC],
];

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    FIXTURE_INPUTS
        .iter()
        .map(|input| {
            vec![
                pack_words(&input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>()),
                vec![0u8; 256 * 4],
            ]
        })
        .collect()
}

const fn build_hist<const N: usize>(input: &[u8; N]) -> [u8; 1024] {
    let mut out = [0u8; 1024];
    let mut i = 0;
    while i < N {
        let b = input[i] as usize;
        let base = b * 4;
        let count = (out[base] as u32)
            | ((out[base + 1] as u32) << 8)
            | ((out[base + 2] as u32) << 16)
            | ((out[base + 3] as u32) << 24);
        let next = count + 1;
        out[base] = next as u8;
        out[base + 1] = (next >> 8) as u8;
        out[base + 2] = (next >> 16) as u8;
        out[base + 3] = (next >> 24) as u8;
        i += 1;
    }
    out
}

const EXPECTED_ENCODEX_HIST_0: [u8; 1024] = build_hist(b"Hello");
const EXPECTED_ENCODEX_HIST_1: [u8; 1024] = build_hist(&[0xC3, 0xA9, 0xC3, 0xA9, b'!']);
const EXPECTED_ENCODEX_HIST_2: [u8; 1024] = build_hist(&[0x00, 0x00, 0x00, 0x41, 0x42]);
const EXPECTED_ENCODEX_HIST_3: [u8; 1024] = build_hist(&[0xE9, 0xE8, 0xEA, 0xEB, 0xEC]);

const EXPECTED_ENCODEX_ENC_0: [u8; 4] = [0, 0, 0, 0];
const EXPECTED_ENCODEX_ENC_1: [u8; 4] = [1, 0, 0, 0];
const EXPECTED_ENCODEX_ENC_2: [u8; 4] = [2, 0, 0, 0];
const EXPECTED_ENCODEX_ENC_3: [u8; 4] = [4, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || encodex_gpu("input", "output", 5),
        Some(fixture_inputs),
        Some(|| {
            vec![
                vec![
                    EXPECTED_ENCODEX_HIST_0.to_vec(),
                    EXPECTED_ENCODEX_ENC_0.to_vec(),
                ],
                vec![
                    EXPECTED_ENCODEX_HIST_1.to_vec(),
                    EXPECTED_ENCODEX_ENC_1.to_vec(),
                ],
                vec![
                    EXPECTED_ENCODEX_HIST_2.to_vec(),
                    EXPECTED_ENCODEX_ENC_2.to_vec(),
                ],
                vec![
                    EXPECTED_ENCODEX_HIST_3.to_vec(),
                    EXPECTED_ENCODEX_ENC_3.to_vec(),
                ],
            ]
        }),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::{bytes_to_u32, decode_u32_one, eval_bytes};

    fn histogram(input: &[u8]) -> (Vec<u32>, u32) {
        let program = encodex_gpu("input", "output", input.len() as u32);
        let input_words: Vec<u32> = if input.is_empty() {
            vec![0]
        } else {
            input.iter().map(|&b| u32::from(b)).collect()
        };
        let outputs = eval_bytes(
            "encodex_gpu",
            &program,
            vec![pack_words(&input_words), vec![0u8; 256 * 4], vec![0u8; 4]],
        );
        (bytes_to_u32(&outputs[0]), decode_u32_one(&outputs[1]))
    }

    fn encodex_reference(input: &[u8]) -> u32 {
        let histogram = vyre_reference::composition_witness::byte_histogram_witness(input);
        vyre_reference::composition_witness::encoding_classify_histogram_witness(
            &histogram,
            input.len() as u32,
        )
    }

    /// WHY: registration literals must stay synchronized with independent IR execution.
    #[test]
    fn registration_outputs_match_reference_execution() {
        let expected_encodings = [
            EXPECTED_ENCODEX_ENC_0,
            EXPECTED_ENCODEX_ENC_1,
            EXPECTED_ENCODEX_ENC_2,
            EXPECTED_ENCODEX_ENC_3,
        ];
        for (input, expected) in FIXTURE_INPUTS.iter().zip(expected_encodings) {
            let (_, actual) = histogram(input);
            assert_eq!(actual.to_le_bytes(), expected);
        }
    }

    #[test]
    fn ascii_detected() {
        let (histogram, enc_id) = histogram(b"Hello");
        assert_eq!(histogram[72], 1);
        assert_eq!(histogram[101], 1);
        assert_eq!(histogram[108], 2);
        assert_eq!(histogram[111], 1);
        assert_eq!(enc_id, ENC_ASCII);
    }

    #[test]
    fn utf8_detected() {
        // é encoded as UTF-8 = 0xC3 0xA9
        let (histogram, enc_id) = histogram(&[0xC3, 0xA9, 0xC3, 0xA9]);
        assert_eq!(histogram[0xC3], 2);
        assert_eq!(histogram[0xA9], 2);
        assert_eq!(enc_id, ENC_UTF8);
    }

    #[test]
    fn high_null_guesses_utf16le() {
        let (histogram, enc_id) = histogram(&[0x00, 0x00, 0x00, 0x41]);
        assert_eq!(histogram[0x00], 3);
        assert_eq!(histogram[0x41], 1);
        assert_eq!(enc_id, ENC_UTF16LE);
    }

    #[test]
    fn iso8859_1_detected() {
        let (histogram, enc_id) = histogram(&[0xE9, 0xE8, 0xEA]);
        assert_eq!(histogram[0xE9], 1);
        assert_eq!(histogram[0xE8], 1);
        assert_eq!(histogram[0xEA], 1);
        assert_eq!(enc_id, ENC_ISO8859_1);
    }

    #[test]
    fn empty_input_is_ascii() {
        let (histogram, enc_id) = histogram(b"");
        assert!(histogram.iter().all(|&v| v == 0));
        assert_eq!(enc_id, ENC_ASCII);
    }

    #[test]
    fn reference_gpu_parity() {
        let inputs: Vec<&[u8]> = vec![
            b"Hello world",
            &[0xC3, 0xA9],
            &[0x00, 0x00, 0x41],
            &[0xE9, 0xE8],
            b"Pure ASCII text here",
        ];
        for input in inputs {
            let (_, gpu_id) = histogram(input);
            let reference_id = encodex_reference(input);
            assert_eq!(
                gpu_id, reference_id,
                "GPU/reference mismatch for input {:?}: gpu={} reference={}",
                input, gpu_id, reference_id
            );
        }
    }

    #[test]
    fn generic_default_names_are_family_scoped() {
        let program = encodex_gpu("input", "output", 8);
        assert_eq!(
            program.buffers()[0].name(),
            fixed_name(FAMILY_PREFIX, "input")
        );
        assert_eq!(
            program.buffers()[2].name(),
            fixed_name(FAMILY_PREFIX, "encoding_id")
        );
    }
}
