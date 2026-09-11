use super::*;

#[test]
fn signature_bytes_extractor_sets_flag() {
    let sig = Signature::bytes_extractor(&[], &[], &[]);
    assert!(sig.bytes_extraction);
}
