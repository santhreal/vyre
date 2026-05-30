//! Adversarial oracle tests for `text::utf8_validate` classification.

use vyre_primitives::text::utf8_validate::{
    reference_utf8_validate, utf8_validate, UTF8_ASCII, UTF8_CONT, UTF8_INVALID, UTF8_LEAD_2,
    UTF8_LEAD_4,
};
use vyre_reference::value::Value;

fn run_program(source: &[u8]) -> Vec<u32> {
    let n = source.len();
    let program = utf8_validate("source", "classes", n as u32);
    let cap = n.max(1);
    let mut input_bytes = Vec::with_capacity(cap * 4);
    for &b in source {
        input_bytes.extend_from_slice(&(b as u32).to_le_bytes());
    }
    while input_bytes.len() < cap * 4 {
        input_bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let zero_classes = vec![0u8; cap * 4];
    let outputs =
        vyre_reference::reference_eval(&program, &[Value::from(input_bytes), Value::from(zero_classes)])
            .expect("Fix: utf8_validate reference evaluation must succeed");
    let out_bytes = outputs[0].to_bytes();
    let mut out_u32s = Vec::with_capacity(cap);
    for chunk in out_bytes.chunks_exact(4) {
        out_u32s.push(u32::from_le_bytes(chunk.try_into().expect("Fix: u32 chunk conversion failed")));
    }
    out_u32s.truncate(n);
    out_u32s
}

#[test]
fn utf8_validate_hostile_corpus() {
    let cases: &[(&[u8], &[u32])] = &[
        (b"", &[]),
        (b"a", &[UTF8_ASCII]),
        (b"\xff", &[UTF8_INVALID]),
        (b"\xc2\x80", &[UTF8_LEAD_2, UTF8_CONT]),
        (b"\xf0\x90\x80\x80", &[UTF8_LEAD_4, UTF8_CONT, UTF8_CONT, UTF8_CONT]),
    ];
    for (idx, (source, expected)) in cases.iter().enumerate() {
        assert_eq!(
            reference_utf8_validate(source),
            *expected,
            "Fix: utf8_validate oracle mismatch on case {idx}"
        );
        assert_eq!(
            run_program(source),
            *expected,
            "Fix: utf8_validate compiled program mismatch on case {idx}"
        );
    }
}
