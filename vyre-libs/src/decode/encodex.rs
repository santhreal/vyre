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

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};
use crate::text::byte_histogram_256_child;
use crate::text::encoding_classify_child;
use crate::text::{ENC_ASCII, ENC_ISO8859_1, ENC_UTF16LE, ENC_UTF8};

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

/// Host-side reference that mirrors the GPU heuristics.
///
/// Computes the same 256-bin histogram and applies the identical
/// classification rules so the host oracle and `encodex_gpu` agree on
/// every fixture input.
///
/// Gated like every other host oracle under `decode/`. It executes the
/// classification on the CPU, so a release consumer of this crate must not
/// compile it, and nothing outside a parity check may call it.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn encodex_reference(input: &[u8]) -> u32 {
    let histogram = crate::text::reference_byte_histogram(input);
    crate::text::classify_from_histogram(&histogram, input.len() as u32)
}

// ---------------------------------------------------------------------------
// Fixtures & harness
// ---------------------------------------------------------------------------

/// One registered case: the input bytes, the histogram bins it fills, and the
/// encoding id the classifier answers.
///
/// The expected side is recorded, not recomputed. Deriving it by calling the
/// host oracle made every build of this crate compile the oracle, and an
/// expectation produced by a twin of the code under test agrees with that code
/// by construction rather than by being right.
struct FixtureCase {
    input: &'static [u8],
    bins: &'static [(u8, u32)],
    encoding: u32,
}

const FIXTURE_CASES: &[FixtureCase] = &[
    FixtureCase {
        input: b"Hello",
        bins: &[(b'H', 1), (b'e', 1), (b'l', 2), (b'o', 1)],
        encoding: ENC_ASCII,
    },
    FixtureCase {
        input: &[0xC3, 0xA9, 0xC3, 0xA9, b'!'],
        bins: &[(b'!', 1), (0xA9, 2), (0xC3, 2)],
        encoding: ENC_UTF8,
    },
    FixtureCase {
        input: &[0x00, 0x00, 0x00, 0x41, 0x42],
        bins: &[(0x00, 3), (0x41, 1), (0x42, 1)],
        encoding: ENC_UTF16LE,
    },
    FixtureCase {
        input: &[0xE9, 0xE8, 0xEA, 0xEB, 0xEC],
        bins: &[(0xE8, 1), (0xE9, 1), (0xEA, 1), (0xEB, 1), (0xEC, 1)],
        encoding: ENC_ISO8859_1,
    },
];

/// Expand a sparse bin list into the packed 256-bin histogram buffer.
fn packed_histogram(bins: &[(u8, u32)]) -> Vec<u8> {
    let mut packed = vec![0u8; 256 * 4];
    for &(byte, count) in bins {
        let slot = usize::from(byte) * 4;
        packed[slot..slot + 4].copy_from_slice(&count.to_le_bytes());
    }
    packed
}

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    FIXTURE_CASES
        .iter()
        .map(|case| {
            vec![
                pack_words(&case.input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>()),
                vec![0u8; 256 * 4],
            ]
        })
        .collect()
}

fn fixture_outputs() -> Vec<Vec<Vec<u8>>> {
    FIXTURE_CASES
        .iter()
        .map(|case| {
            vec![
                packed_histogram(case.bins),
                case.encoding.to_le_bytes().to_vec(),
            ]
        })
        .collect()
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || encodex_gpu("input", "output", 5),
        Some(fixture_inputs),
        Some(fixture_outputs),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn run(input: &[u8]) -> (Vec<u32>, u32) {
        let program = encodex_gpu("input", "output", input.len() as u32);
        let input_words = if input.is_empty() {
            vec![0]
        } else {
            input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>()
        };
        let inputs = vec![
            Value::from(pack_words(&input_words)),
            Value::from(vec![0u8; 256 * 4]),
            Value::from(vec![0u8; 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: encodex must run; restore this invariant before continuing.");
        let histogram = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        let enc_id = u32::from_le_bytes([
            outputs[1].to_bytes()[0],
            outputs[1].to_bytes()[1],
            outputs[1].to_bytes()[2],
            outputs[1].to_bytes()[3],
        ]);
        (histogram, enc_id)
    }

    #[test]
    fn ascii_detected() {
        let (histogram, enc_id) = run(b"Hello");
        assert_eq!(histogram[72], 1);
        assert_eq!(histogram[101], 1);
        assert_eq!(histogram[108], 2);
        assert_eq!(histogram[111], 1);
        assert_eq!(enc_id, ENC_ASCII);
    }

    #[test]
    fn utf8_detected() {
        // é encoded as UTF-8 = 0xC3 0xA9
        let (histogram, enc_id) = run(&[0xC3, 0xA9, 0xC3, 0xA9]);
        assert_eq!(histogram[0xC3], 2);
        assert_eq!(histogram[0xA9], 2);
        assert_eq!(enc_id, ENC_UTF8);
    }

    #[test]
    fn high_null_guesses_utf16le() {
        let (histogram, enc_id) = run(&[0x00, 0x00, 0x00, 0x41]);
        assert_eq!(histogram[0x00], 3);
        assert_eq!(histogram[0x41], 1);
        assert_eq!(enc_id, ENC_UTF16LE);
    }

    #[test]
    fn iso8859_1_detected() {
        let (histogram, enc_id) = run(&[0xE9, 0xE8, 0xEA]);
        assert_eq!(histogram[0xE9], 1);
        assert_eq!(histogram[0xE8], 1);
        assert_eq!(histogram[0xEA], 1);
        assert_eq!(enc_id, ENC_ISO8859_1);
    }

    #[test]
    fn empty_input_is_ascii() {
        let (histogram, enc_id) = run(b"");
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
            let (_, gpu_id) = run(input);
            let cpu_id = encodex_reference(input);
            assert_eq!(
                gpu_id, cpu_id,
                "GPU/CPU mismatch for input {:?}: gpu={} cpu={}",
                input, gpu_id, cpu_id
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
