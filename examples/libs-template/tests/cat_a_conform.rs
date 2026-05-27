//! Cat-A conformance harness for `{{crate_name}}`.
//!
//! Every op in this crate ships a byte-identity witness test here.
//! Follow `AUTHORING.md` in vyre-libs for the pattern.

use {{crate_name_snake}}::example_op;
use vyre_reference::value::Value;

fn u32_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

fn decode_u32_words(bytes: &[u8]) -> Vec<u32> {
    vyre_primitives::wire::decode_u32_le_bytes_all(bytes)
}

#[test]
fn example_op_adds_one_elementwise() {
    let program = example_op("input", "output", 4);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(&[1, 2, 3, 4])),
            Value::from(vec![0u8; 16]),
        ],
    )
    .expect("example_op must execute");
    let got = decode_u32_words(&outputs[0].to_bytes());
    assert_eq!(got, vec![2, 3, 4, 5]);
}
