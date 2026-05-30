//! Adversarial oracle tests for `text::char_class` reference mapping.

use vyre_primitives::text::char_class::{char_class, reference_char_class};
use vyre_reference::value::Value;

fn run_program(source: &[u8], table: &[u32; 256]) -> Vec<u32> {
    let n = source.len();
    let program = char_class("source", "classified", n as u32);
    let cap = n.max(1);
    let mut input_bytes = Vec::with_capacity(cap * 4);
    for &b in source {
        input_bytes.extend_from_slice(&(b as u32).to_le_bytes());
    }
    while input_bytes.len() < cap * 4 {
        input_bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let mut table_bytes = Vec::with_capacity(256 * 4);
    for &t in table {
        table_bytes.extend_from_slice(&t.to_le_bytes());
    }
    let zero_classified = vec![0u8; cap * 4];
    let outputs =
        vyre_reference::reference_eval(
            &program,
            &[Value::from(input_bytes), Value::from(table_bytes), Value::from(zero_classified)]
        )
        .expect("Fix: char_class reference evaluation must succeed");
    let out_bytes = outputs[0].to_bytes();
    let mut out_u32s = Vec::with_capacity(cap);
    for chunk in out_bytes.chunks_exact(4) {
        out_u32s.push(u32::from_le_bytes(chunk.try_into().expect("Fix: u32 chunk conversion failed")));
    }
    out_u32s.truncate(n);
    out_u32s
}

#[test]
fn char_class_hostile_corpus_table_driven() {
    const ZEROS: [u32; 256] = [0u32; 256];
    const MAXES: [u32; 256] = [0xffff_ffffu32; 256];
    let cases: &[(&[u8], &[u32; 256], &[u32])] = &[
        (b"", &ZEROS, &[]),
        (b"\x00", &ZEROS, &[0]),
        (b"\xff", &MAXES, &[0xffff_ffff]),
        (b"vyre", &ZEROS, &[0, 0, 0, 0]),
        (b"\x00\xff\xfe", &MAXES, &[0xffff_ffff, 0xffff_ffff, 0xffff_ffff]),
    ];
    for (idx, (source, table, expected)) in cases.iter().enumerate() {
        let got = reference_char_class(source, table);
        assert_eq!(
            got, *expected,
            "Fix: char_class oracle mismatch on hostile case {idx} (len={})",
            source.len()
        );
        assert_eq!(
            run_program(source, table),
            *expected,
            "Fix: char_class compiled program mismatch on hostile case {idx} (len={})",
            source.len()
        );
    }
}
