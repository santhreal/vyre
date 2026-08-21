//! Expected output bytes shared by several conformance registrations.
//!
//! A witness answer written twice is a witness that can disagree with itself,
//! so the shared ones are stated here once.

pub(crate) fn u32_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

pub(crate) const MATMUL_2X2_EXPECTED_BYTES: [u8; 16] = [
    0x13, 0x00, 0x00, 0x00, // 19
    0x16, 0x00, 0x00, 0x00, // 22
    0x2b, 0x00, 0x00, 0x00, // 43
    0x32, 0x00, 0x00, 0x00, // 50
];

#[cfg(test)]
#[must_use]
pub(crate) fn matmul_2x2_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![MATMUL_2X2_EXPECTED_BYTES.to_vec()]]
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

#[cfg(test)]
pub(crate) fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    vyre_primitives::wire::decode_f32_le_bytes_all(bytes)
}

#[cfg(test)]
pub(crate) fn decode_f32_one(bytes: &[u8]) -> f32 {
    match try_decode_f32_one(bytes) {
        Ok(value) => value,
        Err(_) => f32::NAN,
    }
}

#[cfg(test)]
pub(crate) fn try_decode_f32_one(bytes: &[u8]) -> Result<f32, String> {
    vyre_primitives::wire::read_f32_le_word(bytes, 0, "f32 scalar fixture output")
}

#[cfg(test)]
pub(crate) fn decode_u32_one(bytes: &[u8]) -> u32 {
    match try_decode_u32_one(bytes) {
        Ok(value) => value,
        Err(_) => u32::MAX,
    }
}

#[cfg(test)]
pub(crate) fn try_decode_u32_one(bytes: &[u8]) -> Result<u32, String> {
    vyre_primitives::wire::read_u32_le_word(bytes, 0, "u32 scalar fixture output")
}

#[cfg(test)]
pub(crate) fn bytes_to_u32(slice: &[u8]) -> Vec<u32> {
    vyre_primitives::wire::decode_u32_le_bytes_all(slice)
}

/// Run a program through the reference interpreter and hand back raw buffers.
///
/// Every module test that wanted a reference answer used to pack its own
/// buffers, call `reference_eval`, and decode the result, so the same eight
/// lines existed once per module under a local `run`. Two of those copies had
/// already drifted into passing a differently sized output buffer than the
/// program declared. This is the one place the call is made.
///
/// `buffers` is the complete argument list in declaration order, outputs
/// included: a zeroed vector of the right length is what the interpreter
/// writes into, and a zero-length one is a real argument, not an omission.
#[cfg(test)]
pub(crate) fn eval_bytes(
    label: &str,
    program: &vyre_foundation::ir::Program,
    buffers: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    try_eval_bytes(program, buffers).unwrap_or_else(|error| {
        panic!("Fix: {label} program must execute in the reference interpreter: {error:?}")
    })
}

/// Run a program that is expected to trap, and hand back the refusal.
///
/// A trap contract asserts on the error, so it cannot go through
/// [`eval_bytes`], which panics. Both share this body so the interpreter is
/// still reached from one place.
#[cfg(test)]
pub(crate) fn try_eval_bytes(
    program: &vyre_foundation::ir::Program,
    buffers: Vec<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, vyre_reference::ReferenceError> {
    let values: Vec<vyre_reference::value::Value> = buffers
        .into_iter()
        .map(vyre_reference::value::Value::from)
        .collect();
    Ok(vyre_reference::reference_eval(program, &values)?
        .iter()
        .map(|value| value.to_bytes())
        .collect())
}

/// Run a program with the interpreter's lanes in declaration order, or
/// reversed.
///
/// A determinism probe runs the same program both ways and compares the
/// buffers. Which entry point runs is the thing under test there, so the
/// probe cannot go through [`eval_bytes`], which only knows the default
/// order.
#[cfg(test)]
pub(crate) fn eval_bytes_lane_order(
    label: &str,
    program: &vyre_foundation::ir::Program,
    buffers: Vec<Vec<u8>>,
    reversed: bool,
) -> Vec<Vec<u8>> {
    let values: Vec<vyre_reference::value::Value> = buffers
        .into_iter()
        .map(vyre_reference::value::Value::from)
        .collect();
    let results = if reversed {
        vyre_reference::reference_eval_lane_reversed(program, &values)
    } else {
        vyre_reference::reference_eval(program, &values)
    };
    results
        .unwrap_or_else(|error| {
            panic!("Fix: {label} program must execute in the reference interpreter: {error:?}")
        })
        .iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Run a program whose arguments and one output are all f32.
///
/// `output_len` is the element count the program declares for its output, not
/// the input length: an operation that reduces or expands declares a different
/// one, and sizing the buffer from an input is how a reduction test ends up
/// reading a buffer the program never filled.
#[cfg(test)]
pub(crate) fn eval_f32(
    label: &str,
    program: &vyre_foundation::ir::Program,
    inputs: &[&[f32]],
    output_len: usize,
) -> Vec<f32> {
    let mut buffers: Vec<Vec<u8>> = inputs.iter().map(|input| f32_bytes(input)).collect();
    buffers.push(vec![0u8; output_len * 4]);
    decode_f32(&eval_bytes(label, program, buffers)[0])
}

/// Run a program whose arguments and one output are all u32.
#[cfg(test)]
pub(crate) fn eval_u32(
    label: &str,
    program: &vyre_foundation::ir::Program,
    inputs: &[&[u32]],
    output_len: usize,
) -> Vec<u32> {
    let mut buffers: Vec<Vec<u8>> = inputs.iter().map(|input| u32_bytes(input)).collect();
    buffers.push(vec![0u8; output_len * 4]);
    bytes_to_u32(&eval_bytes(label, program, buffers)[0])
}

/// Run a one-in one-out f32 program through the reference interpreter.
///
/// The argument shape is the one every elementwise and per-row nn program under
/// test declares: the packed input, then a zeroed output buffer of the same
/// length. `label` names the program in the failure.
#[cfg(test)]
pub(crate) fn eval_f32_unary(
    label: &str,
    input: &[f32],
    program: &vyre_foundation::ir::Program,
) -> Vec<f32> {
    eval_f32(label, program, &[input], input.len())
}

/// Compare a tiled f32 program against its scalar reference, lane by lane.
///
/// The lane counts are compared first: a tiled program that wrote fewer lanes
/// than the reference would otherwise pass the elementwise comparison, because
/// zipping two sequences stops at the shorter one.
#[cfg(test)]
pub(crate) fn assert_tiled_matches_reference(
    label: &str,
    input: &[f32],
    tolerance: f32,
    tiled: &vyre_foundation::ir::Program,
    reference: &vyre_foundation::ir::Program,
) {
    let actual = eval_f32_unary(label, input, tiled);
    let expected = eval_f32_unary(label, input, reference);
    assert_eq!(
        actual.len(),
        expected.len(),
        "Fix: {label} must write the same lane count as its reference."
    );
    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= tolerance,
            "{label} mismatch at lane {idx}: tiled={lhs:?} reference={rhs:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2_expected_bytes_identity() {
        let constructed = u32_bytes(&[19, 22, 43, 50]);
        assert_eq!(constructed, MATMUL_2X2_EXPECTED_BYTES);
    }

    #[test]
    fn test_round_trip() {
        let original = vec![1, 2, 3, 0xFFFFFFFF, 0x12345678];
        let bytes = u32_bytes(&original);
        let back = bytes_to_u32(&bytes);
        assert_eq!(original, back);
    }

    #[test]
    fn test_empty_input() {
        let original: Vec<u32> = vec![];
        let bytes = u32_bytes(&original);
        assert!(bytes.is_empty());
        let back = bytes_to_u32(&bytes);
        assert!(back.is_empty());
    }

    #[test]
    fn test_f32_bit_exact_pack() {
        let bytes = f32_bytes(&[1.0, -0.0, f32::INFINITY, f32::NAN]);
        let unpacked =
            vyre_primitives::wire::unpack_f32_slice(&bytes, 4, "test_f32_bit_exact_pack")
                .expect("Fix: f32 test fixture pack must round-trip.");
        assert_eq!(unpacked[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(unpacked[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(unpacked[2].to_bits(), f32::INFINITY.to_bits());
        assert!(unpacked[3].is_nan());
    }
}
