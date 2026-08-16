//! `prover` contracts over the public `vyre_conform` surface.

use vyre_conform::prover::*;

#[test]
fn xor_is_commutative_and_associative() {
    let w: Vec<u32> = (0..8).collect();
    assert_eq!(
        LawProver::verify_commutative(|a, b| a ^ b, &w),
        LawVerdict::Holds
    );
    assert_eq!(
        LawProver::verify_associative(|a, b| a ^ b, &w),
        LawVerdict::Holds
    );
    assert_eq!(
        LawProver::verify_identity(|a, b| a ^ b, 0, &w),
        LawVerdict::Holds
    );
}

#[test]
fn sub_is_not_commutative_reports_correct_counterexample() {
    let w: Vec<u32> = vec![1, 2, 3];
    let verdict = LawProver::verify_commutative(|a, b| a.wrapping_sub(b), &w);
    match verdict {
        LawVerdict::CommutativeFails { a, b, ab, ba } => {
            assert_ne!(
                ab, ba,
                "counterexample must have ab != ba; got a={a} b={b} ab={ab} ba={ba}"
            );
            assert_eq!(ab, a.wrapping_sub(b), "ab must equal f(a,b)");
            assert_eq!(ba, b.wrapping_sub(a), "ba must equal f(b,a)");
        }
        other => panic!("expected CommutativeFails, got {other:?}"),
    }
}

#[test]
fn empty_witnesses_return_no_witnesses_not_holds() {
    let empty: &[u32] = &[];
    assert_eq!(
        LawProver::verify_commutative(|a, b| a.wrapping_sub(b), empty),
        LawVerdict::NoWitnesses,
        "verify_commutative with empty witnesses must return NoWitnesses, not Holds"
    );
    assert_eq!(
        LawProver::verify_associative(|a, b| a.wrapping_sub(b), empty),
        LawVerdict::NoWitnesses,
        "verify_associative with empty witnesses must return NoWitnesses, not Holds"
    );
    assert_eq!(
        LawProver::verify_identity(|a, b| a ^ b, 0, empty),
        LawVerdict::NoWitnesses,
        "verify_identity with empty witnesses must return NoWitnesses, not Holds"
    );
}
