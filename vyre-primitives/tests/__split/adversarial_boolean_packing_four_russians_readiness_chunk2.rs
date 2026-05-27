#[test]
fn four_russians_ir_apply_lut_matches_cpu_reference() {
    let lhs = [0xFF00_FF00u32, 0x0F0F_0F0F];
    let rhs = [0xF0F0_F0F0u32, 0xFFFF_0000];
    let lut = binary_byte_lut(BooleanTileOp::And);
    let program = four_russians_apply_byte_lut("lhs", "rhs", "lut", "out", lhs.len() as u32);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(&lhs)),
            Value::from(u32_bytes(&rhs)),
            Value::from(u32_bytes(&lut)),
            Value::from(vec![0u8; lhs.len() * 4]),
        ],
    )
    .expect("four_russians_apply_byte_lut must execute");

    assert_eq!(
        outputs[0].to_bytes(),
        u32_bytes(&four_russians_cpu_ref(&lhs, &rhs, &lut))
    );
}

// ---------------------------------------------------------------------------
// Adversarial boundary cases
// ---------------------------------------------------------------------------

#[test]
fn all_ops_on_maximum_word_count() {
    // 128 bits = 4 words. Stress the widest common case.
    let a = vec![0x1234_5678u32, 0x9ABC_DEF0, 0x0F0F_0F0F, 0xF0F0_F0F0];
    let b = vec![0xF0F0_F0F0u32, 0x0F0F_0F0F, 0x9ABC_DEF0, 0x1234_5678];

    let and_got = bitset_and_ref(&a, &b);
    let or_got = bitset_or_ref(&a, &b);
    let xor_got = bitset_xor_ref(&a, &b);

    for i in 0..4 {
        assert_eq!(and_got[i], a[i] & b[i]);
        assert_eq!(or_got[i], a[i] | b[i]);
        assert_eq!(xor_got[i], a[i] ^ b[i]);
    }
}

#[test]
fn not_is_involution() {
    let a = vec![0xA5A5_A5A5u32, 0x5A5A_5A5A];
    let not_not_a = bitset_not_ref(&bitset_not_ref(&a));
    assert_eq!(not_not_a, a, "NOT(NOT(x)) must equal x");
}

#[test]
fn xor_is_self_inverse() {
    let a = vec![0xDEAD_BEEFu32];
    let b = vec![0xDEAD_BEEFu32];
    assert_eq!(bitset_xor_ref(&a, &b), vec![0], "x ^ x must be 0");
}

#[test]
fn or_is_idempotent() {
    let a = vec![0xCAFE_BABEu32];
    assert_eq!(bitset_or_ref(&a, &a), a, "x | x must equal x");
}

#[test]
fn and_is_idempotent() {
    let a = vec![0xCAFE_BABEu32];
    assert_eq!(bitset_and_ref(&a, &a), a, "x & x must equal x");
}

#[test]
fn de_morgan_law_holds_at_word_level() {
    let a = vec![0xF0F0_F0F0u32];
    let b = vec![0x0F0F_0F0Fu32];
    // NOT(a AND b) == NOT(a) OR NOT(b)
    let lhs = bitset_not_ref(&bitset_and_ref(&a, &b));
    let rhs = bitset_or_ref(&bitset_not_ref(&a), &bitset_not_ref(&b));
    assert_eq!(lhs, rhs, "De Morgan's law must hold for word-level bitsets");
}
