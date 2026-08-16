//! Wire helpers every `vyre-libs` contract test packs its oracle buffers with.
//!
//! `vyre_primitives::wire` already owns the little-endian packers and decoders,
//! so a test that writes its own `flat_map(to_le_bytes)` loop is a second copy
//! of a shipped primitive. The BF16 rounding has no production owner because
//! only the typed contracts need it, so it is owned here.
#![allow(unused_imports, unused_macros)]

use vyre_foundation::ir::Program;
use vyre_primitives::wire::decode_u16_le_bytes_all;
use vyre_reference::value::Value;

pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

pub(crate) use vyre_primitives::wire::pack_u32_slice as bytes_from_words;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;

pub(crate) use vyre_primitives::wire::pack_f32_slice as f32_bytes;

pub(crate) use vyre_primitives::wire::decode_f32_le_bytes_all as f32_words;

pub(crate) use vyre_primitives::wire::pack_u16_slice as u16_bytes;

pub(crate) use vyre_primitives::wire::decode_u16_le_bytes_all as u16_words;

/// F32 words from an oracle output value.
pub(crate) fn f32_words_of(value: &Value) -> Vec<f32> {
    f32_words(&value.to_bytes())
}

/// U16 words from an oracle output value, the carrier for a BF16 or F16 lane.
pub(crate) fn u16_words_of(value: &Value) -> Vec<u16> {
    decode_u16_le_bytes_all(&value.to_bytes())
}

/// Round `value` to BF16, breaking ties toward even, the rounding the typed
/// kernels do when they narrow an F32 lane.
pub(crate) fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// BF16 wire bytes for `values`.
pub(crate) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    u16_bytes(&values.iter().copied().map(bf16_word).collect::<Vec<_>>())
}

/// The one-workgroup over-fire dispatch floor shared by every over-fire gate: the
/// largest declared buffer element count plus one whole workgroup of lanes, the
/// realistic worst case a whole-workgroup GPU dispatch produces past the logical
/// element count. ONE home so no two over-fire gates can drift.
pub(crate) fn overfire_grid(program: &Program) -> u32 {
    let workgroup_lanes = program.workgroup_size()[0].max(1);
    let max_count = program
        .buffers()
        .iter()
        .map(vyre_foundation::ir::BufferDecl::count)
        .max()
        .unwrap_or(0);
    max_count.saturating_add(workgroup_lanes)
}

pub(crate) fn reference_eval_idoms(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
) -> Vec<u32> {
    let to_bytes = |w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();

    let values: Vec<Value> = vec![
        Value::from(to_bytes(edge_offsets)),
        Value::from(to_bytes(edge_targets)),
        Value::from(to_bytes(pred_offsets)),
        Value::from(to_bytes(pred_targets)),
        Value::from(to_bytes(&vec![0u32; node_count as usize])),
        Value::from(to_bytes(&vec![0u32; node_count as usize])),
    ];

    let outputs = vyre_reference::reference_eval(program, &values)
        .expect("dominator-tree reference program must evaluate");
    let bytes = outputs[0].to_bytes();
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("u32 output chunk has four bytes")))
        .collect()
}

macro_rules! adversarial_unary_vec_cases {
    ($($name:ident: $input:expr => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let input = $input;
                let expected = $expected;
                let actual = cpu_ref(&input);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}

macro_rules! adversarial_binary_vec_cases {
    ($($name:ident: $lhs:expr, $rhs:expr => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let lhs = $lhs;
                let rhs = $rhs;
                let expected = $expected;
                let actual = cpu_ref(&lhs, &rhs);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}

macro_rules! adversarial_binary_vec_usize_cases {
    ($($name:ident: $lhs:expr, $rhs:expr, $len:expr => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let lhs = $lhs;
                let rhs = $rhs;
                let len = $len;
                let expected = $expected;
                let actual = cpu_ref(&lhs, &rhs, len);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}

macro_rules! adversarial_vec_u32_cases {
    ($($name:ident: $input:expr, $param:expr => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let input = $input;
                let param = $param;
                let expected = $expected;
                let actual = cpu_ref(&input, param);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}

#[cfg(feature = "go-parser")]
pub(crate) mod go;
