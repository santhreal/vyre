//! Adversarial oracle tests for `text::line_index`.

use vyre_primitives::text::line_index::{line_index, reference_line_index};
use vyre_reference::value::Value;

fn run_program(source: &[u8]) -> Vec<u32> {
    let n = source.len();
    let program = line_index("source", "lines", n as u32);
    let cap = n.max(1);
    let mut input_bytes = Vec::with_capacity(cap * 4);
    for &b in source {
        input_bytes.extend_from_slice(&(b as u32).to_le_bytes());
    }
    while input_bytes.len() < cap * 4 {
        input_bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let zero_lines = vec![0u8; cap * 4];
    let outputs =
        vyre_reference::reference_eval(&program, &[Value::from(input_bytes), Value::from(zero_lines)])
            .expect("Fix: line_index reference evaluation must succeed");
    let out_bytes = outputs[0].to_bytes();
    let mut out_u32s = Vec::with_capacity(cap);
    for chunk in out_bytes.chunks_exact(4) {
        out_u32s.push(u32::from_le_bytes(chunk.try_into().expect("Fix: u32 chunk conversion failed")));
    }
    out_u32s.truncate(n);
    out_u32s
}

#[test]
fn line_index_hostile_corpus() {
    let cases: &[(&[u8], &[u32])] = &[
        (b"", &[]),
        (b"a", &[0]),
        (b"a\nb", &[0, 0, 1]),
        (b"a\r\nb", &[0, 0, 0, 1]),
        (b"\r\n\r\n", &[0, 0, 1, 1]),
    ];
    for (idx, (source, expected)) in cases.iter().enumerate() {
        assert_eq!(
            reference_line_index(source),
            *expected,
            "Fix: line_index oracle mismatch on case {idx}"
        );
        assert_eq!(
            run_program(source),
            *expected,
            "Fix: line_index compiled program mismatch on case {idx}"
        );
    }
}
