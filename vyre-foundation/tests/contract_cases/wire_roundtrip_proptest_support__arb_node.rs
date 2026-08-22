#[path = "wire_roundtrip_proptest__every_expression_variant_roundtrips_in_one_program.rs"]
mod wire_roundtrip_proptest_every_expression_variant_roundtrips_in_one_program;
#[path = "wire_roundtrip_proptest__program_wire_roundtrip_preserves_structure.rs"]
mod wire_roundtrip_proptest_program_wire_roundtrip_preserves_structure;

use super::*;

fn arb_node() -> BoxedStrategy<Node> {
    arb_node_with_depth(3)
}

fn arb_node_with_depth(depth: u32) -> BoxedStrategy<Node> {
    let leaf = arb_statement_leaf(arb_expr);
    if depth == 0 {
        return leaf;
    }
    leaf.prop_recursive(3, 64, 3, move |inner| arb_control_flow(arb_expr, inner))
        .boxed()
}

fn arb_program() -> BoxedStrategy<Program> {
    arb_program_with(arb_node())
}

fn first_replaced(bytes: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    assert_eq!(needle.len(), replacement.len());
    let mut mutated = bytes.to_vec();
    let offset = mutated[40..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| offset + 40)
        .expect("Fix: expected to find the encoded extension id in the wire body");
    mutated[offset..offset + needle.len()].copy_from_slice(replacement);
    mutated
}

fn first_replaced_with_valid_digest(bytes: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let mut mutated = first_replaced(bytes, needle, replacement);
    let digest = blake3::hash(&mutated[40..]);
    mutated[8..40].copy_from_slice(digest.as_bytes());
    mutated
}

fn first_let_expr(program: &Program) -> &Expr {
    match top_level_body(program).first() {
        Some(Node::Let { value, .. }) => value,
        other => panic!("Fix: expected first node to be a let binding, got {other:?}"),
    }
}

fn top_level_body(program: &Program) -> &[Node] {
    match program.entry().first() {
        Some(Node::Region { body, .. }) => body.as_slice(),
        _ => program.entry(),
    }
}

/// Mirror of `serial::wire::encode::put_expr::canonical_f32_bits` so
/// this crate's tests can compare against the wire's canonical form
/// without pulling the private encoder helper.
///
/// Wire canonicalization is more aggressive than
/// `vyre_reference::ieee754::canonical_f32`: BOTH subnormal signs
/// AND -0.0 flush to +0.0. NaN payloads collapse to the single
/// positive qNaN (0x7FC0_0000).
fn canonicalize_f32(value: f32) -> f32 {
    if value.is_nan() {
        return f32::from_bits(0x7FC0_0000);
    }
    if value.is_subnormal() {
        return 0.0_f32;
    }
    if value.to_bits() == (-0.0_f32).to_bits() {
        return 0.0_f32;
    }
    value
}
