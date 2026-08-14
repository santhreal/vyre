//! Elementwise parity-gate scaffolding shared by the backend driver crates.
//!
//! A parity gate dispatches one small program on a real device and asserts the
//! bytes read back against a host reference. What differs between such gates is
//! the operation, the operand table, and the reference; what does not differ is
//! the little-endian packing, the single-thread elementwise program shape, the
//! zero-initialised `ReadWrite` output at binding 0, and the decode of the
//! returned buffer. That invariant half lived once per gate per backend crate,
//! which is how a fixed packing bug in one copy stayed live in the others.
//!
//! Everything here is expressed against [`VyreBackend`] and
//! [`vyre_foundation::ir`], so it names no target, dialect or driver.
//!
//! Enabled by the `test-fixtures` feature: it is scaffolding, not product code,
//! and a published build should not carry it.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::BackendError;

/// Pack words little-endian.
#[must_use]
pub fn u32_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Decode a device buffer as little-endian `u32` words.
///
/// # Panics
///
/// Panics when `bytes` is not a whole number of words: a partial trailing word
/// means the output buffer was mis-sized, which is the defect a parity gate
/// exists to catch, not something to round away.
#[must_use]
pub fn u32_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "Fix: a u32 output buffer must be a whole number of 4-byte words; got {} bytes.",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Decode a device buffer as little-endian `u64` words.
///
/// # Panics
///
/// Panics when `bytes` is not a whole number of 8-byte elements.
#[must_use]
pub fn u64_words(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(
        bytes.len() % 8,
        0,
        "Fix: a u64 output buffer must be a whole number of 8-byte elements; got {} bytes.",
        bytes.len()
    );
    bytes
        .chunks_exact(8)
        .map(|element| {
            let low = u32::from_le_bytes([element[0], element[1], element[2], element[3]]);
            let high = u32::from_le_bytes([element[4], element[5], element[6], element[7]]);
            (u64::from(high) << 32) | u64::from(low)
        })
        .collect()
}

/// One read-only input buffer of an elementwise parity program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParityInput {
    /// Buffer name the program loads from.
    pub name: &'static str,
    /// Element type of the buffer.
    pub element: DataType,
    /// Packed contents, one element per lane.
    pub bytes: Vec<u8>,
}

impl ParityInput {
    /// A `u32` input buffer from host words.
    #[must_use]
    pub fn u32_words(name: &'static str, words: &[u32]) -> Self {
        Self {
            name,
            element: DataType::U32,
            bytes: u32_bytes(words),
        }
    }

    /// An input buffer of `element` type whose bytes are already packed.
    #[must_use]
    pub fn packed(name: &'static str, element: DataType, bytes: Vec<u8>) -> Self {
        Self {
            name,
            element,
            bytes,
        }
    }
}

/// Build `out[i] = build(&[load(input_0, i), ..])` for every `i` in `0..count`.
///
/// `out` is the `ReadWrite` storage buffer at binding 0; inputs take bindings
/// `1..` in the order given. Workgroup geometry is a single thread, so the whole
/// lane range is written by one invocation and no lane is left to the
/// scheduler.
///
/// # Panics
///
/// Panics when `build` is handed a different number of loads than there are
/// inputs, which would silently drop an operand.
#[must_use]
pub fn elementwise_program(
    out_element: DataType,
    inputs: &[ParityInput],
    count: u32,
    build: &dyn Fn(&[Expr]) -> Expr,
) -> Program {
    let mut buffers = Vec::with_capacity(inputs.len() + 1);
    buffers.push(
        BufferDecl::storage("out", 0, BufferAccess::ReadWrite, out_element).with_count(count),
    );
    for (offset, input) in inputs.iter().enumerate() {
        let binding = u32::try_from(offset + 1).expect("parity programs declare few buffers");
        buffers.push(
            BufferDecl::storage(
                input.name,
                binding,
                BufferAccess::ReadOnly,
                input.element.clone(),
            )
            .with_count(count),
        );
    }
    let mut body = Vec::with_capacity(count as usize);
    for lane in 0..count {
        let loads = inputs
            .iter()
            .map(|input| Expr::load(input.name, Expr::u32(lane)))
            .collect::<Vec<_>>();
        body.push(Node::store("out", Expr::u32(lane), build(&loads)));
    }
    Program::wrapped(buffers, [1, 1, 1], body)
}

/// How a gate hands a program and its borrowed inputs to its own backend.
///
/// A concrete driver's dispatch entry point is not always a trait method, so the
/// harness takes the call rather than the receiver. That keeps this crate free
/// of any concrete backend type while still owning everything around the call.
pub type ParityDispatch<'a> =
    &'a dyn Fn(&Program, &[&[u8]]) -> Result<Vec<Vec<u8>>, BackendError>;

/// Dispatch a program whose only output is the zero-initialised buffer at
/// binding 0, and return that buffer's bytes.
///
/// `output_bytes` sizes the zero-initialised output, so a gate states the width
/// it expects rather than reconstructing it from the element type at each call.
///
/// # Panics
///
/// Panics when dispatch fails, when the backend returns anything other than one
/// buffer, or when that buffer is not `output_bytes` long. Each is a contract
/// break a parity gate must surface, and `label` names the gate in the message.
#[must_use]
pub fn dispatch_single_output(
    dispatch: ParityDispatch<'_>,
    program: &Program,
    inputs: &[ParityInput],
    output_bytes: usize,
    label: &str,
) -> Vec<u8> {
    let zeroed = vec![0u8; output_bytes];
    let mut slices = Vec::with_capacity(inputs.len() + 1);
    slices.push(zeroed.as_slice());
    for input in inputs {
        slices.push(input.bytes.as_slice());
    }
    let outputs = dispatch(program, &slices)
        .unwrap_or_else(|error| panic!("Fix: the backend must dispatch `{label}`: {error}"));
    assert_eq!(
        outputs.len(),
        1,
        "`{label}` declares one ReadWrite output; the backend returned {} buffer(s).",
        outputs.len()
    );
    assert_eq!(
        outputs[0].len(),
        output_bytes,
        "`{label}` output buffer must be {output_bytes} bytes; got {}.",
        outputs[0].len()
    );
    outputs.into_iter().next().expect("length asserted above")
}

/// Build, dispatch and decode a `u32` elementwise gate in one call.
///
/// The three-buffer `out[i] = build(a[i], b[i])` shape is what every binary
/// `u32` parity gate needs, so it is spelled once here instead of once per gate.
#[must_use]
pub fn u32_binop_parity(
    dispatch: ParityDispatch<'_>,
    build: fn(Expr, Expr) -> Expr,
    pairs: &[(u32, u32)],
    label: &str,
) -> Vec<u32> {
    let lefts = pairs.iter().map(|&(left, _)| left).collect::<Vec<_>>();
    let rights = pairs.iter().map(|&(_, right)| right).collect::<Vec<_>>();
    let inputs = vec![
        ParityInput::u32_words("a", &lefts),
        ParityInput::u32_words("b", &rights),
    ];
    let count = u32::try_from(pairs.len()).expect("parity gates use small operand tables");
    let program = elementwise_program(DataType::U32, &inputs, count, &|loads| {
        build(loads[0].clone(), loads[1].clone())
    });
    let bytes = dispatch_single_output(dispatch, &program, &inputs, pairs.len() * 4, label);
    u32_words(&bytes)
}

#[cfg(test)]
mod tests {
    use super::{elementwise_program, u32_bytes, u32_words, u64_words, ParityInput};
    use vyre_foundation::ir::{BufferAccess, DataType, Expr};

    /// WHY: every gate's operands and every decoded result pass through this
    /// packing. A byte order or width error here would make each gate agree
    /// with a wrong reference rather than fail.
    #[test]
    fn word_packing_round_trips_little_endian() {
        let words = [0u32, 1, 0x8000_0000, u32::MAX, 0x0403_0201];
        assert_eq!(u32_bytes(&words)[16..20], [0x01, 0x02, 0x03, 0x04]);
        assert_eq!(u32_words(&u32_bytes(&words)), words);
    }

    /// WHY: a 64-bit gate reads the high word from the second half of each
    /// element. Swapping the halves is the exact defect that makes a
    /// sign-extension bug look like a passing test.
    #[test]
    fn u64_decode_takes_the_high_word_from_the_upper_half() {
        let bytes = u32_bytes(&[0x0000_0009, 0xFFFF_FFFF]);
        assert_eq!(u64_words(&bytes), [0xFFFF_FFFF_0000_0009]);
    }

    /// WHY: the output must be `ReadWrite` at binding 0 with inputs after it,
    /// because the dispatch helper passes the zeroed output as the first slice.
    /// A reordered binding table would feed operands into the output slot.
    #[test]
    fn output_is_read_write_at_binding_zero_and_inputs_follow() {
        let inputs = vec![
            ParityInput::u32_words("a", &[1, 2]),
            ParityInput::u32_words("b", &[3, 4]),
        ];
        let program = elementwise_program(DataType::U32, &inputs, 2, &|loads| {
            Expr::add(loads[0].clone(), loads[1].clone())
        });
        let buffers = program.buffers();
        assert_eq!(
            buffers
                .iter()
                .map(|buffer| (buffer.name(), buffer.binding()))
                .collect::<Vec<_>>(),
            vec![("out", 0), ("a", 1), ("b", 2)]
        );
        assert_eq!(
            buffers[0].access(),
            BufferAccess::ReadWrite,
            "Fix: the parity output buffer must be writable."
        );
    }

    /// WHY: one invocation must write every lane. If geometry widened, lanes
    /// beyond the first workgroup would be written by threads the program never
    /// declared, and the gate would read back its own zero-init.
    #[test]
    fn every_lane_is_written_by_a_single_invocation() {
        let inputs = vec![ParityInput::u32_words("a", &[1, 2, 3])];
        let program = elementwise_program(DataType::U32, &inputs, 3, &|loads| loads[0].clone());
        assert_eq!(program.workgroup_size(), [1, 1, 1]);
        assert!(program.top_level_region_violation().is_none());
    }
}
