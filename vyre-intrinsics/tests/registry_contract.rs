//! Stable intrinsic catalog and deterministic fixture contracts.

use vyre_foundation::dialect_lookup::Signature;
use vyre_foundation::ir::{Node, Program};
use vyre_intrinsics::harness::{
    all_entries, validate_entries, OpEntry, RegistryError, F32_UNARY_SIGNATURE,
    U32_UNARY_SIGNATURE,
};

fn empty_program() -> Program {
    Program::default()
}

fn entry(id: &'static str, signature: Signature) -> OpEntry {
    OpEntry::new(id, signature, empty_program, None, None)
}

/// Positive: the linked catalog is deterministic, fully signed, and its
/// fixtures are stable across repeated reads.
#[test]
fn canonical_catalog_is_deterministic_and_fixture_complete() {
    let entries = all_entries().collect::<Vec<_>>();
    assert!(!entries.is_empty());
    assert!(entries.windows(2).all(|pair| pair[0].id < pair[1].id));

    for intrinsic in entries {
        let shape = intrinsic.shape().expect("hardware intrinsic shape");
        assert_eq!(intrinsic.signature.inputs.len(), shape.input_buffers as usize);
        assert_eq!(intrinsic.signature.outputs.len(), shape.output_buffers as usize);
        let program = (intrinsic.build)();
        match program.entry().first() {
            Some(Node::Region { generator, .. }) => {
                assert_eq!(generator.as_str(), intrinsic.id);
            }
            _ => panic!("{} must build one canonical intrinsic region", intrinsic.id),
        }

        let inputs = intrinsic.test_inputs.expect("deterministic inputs");
        let expected = intrinsic.expected_output.expect("deterministic outputs");
        assert_eq!(inputs(), inputs(), "{} input fixtures changed", intrinsic.id);
        assert_eq!(
            expected(),
            expected(),
            "{} expected fixtures changed",
            intrinsic.id
        );
    }
}

/// Negative: a second owner for an existing identity is rejected even when it
/// repeats the exact signature.
#[test]
fn duplicate_identity_is_rejected() {
    let first = entry("test.intrinsic", U32_UNARY_SIGNATURE);
    let duplicate = entry("test.intrinsic", U32_UNARY_SIGNATURE);
    assert_eq!(
        validate_entries([&first, &duplicate]),
        Err(RegistryError::DuplicateId {
            id: "test.intrinsic"
        })
    );
}

/// Boundary: an empty extension catalog is valid and requires no special
/// sentinel registration.
#[test]
fn empty_catalog_is_valid() {
    assert_eq!(validate_entries(std::iter::empty()), Ok(()));
}

/// Adversarial: reusing a stable identity while changing the scalar contract
/// fails as a signature mismatch rather than being hidden as a duplicate.
#[test]
fn identity_with_mismatched_signature_is_rejected() {
    let canonical = entry("test.intrinsic", U32_UNARY_SIGNATURE);
    let conflicting = entry("test.intrinsic", F32_UNARY_SIGNATURE);
    assert_eq!(
        validate_entries([&canonical, &conflicting]),
        Err(RegistryError::SignatureMismatch {
            id: "test.intrinsic"
        })
    );
}
