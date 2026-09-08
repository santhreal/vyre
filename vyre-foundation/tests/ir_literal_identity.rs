//! Is a program equal to itself?
//!
//! The optimizer decides it has converged by asking each pass whether the
//! program changed, and a pass answers by comparing the tree it produced with
//! the tree it received. That answer is only meaningful if a program is equal
//! to itself, which IEEE equality denies for any program holding a NaN literal:
//! `NaN != NaN`, so every rebuild of the tree reads as a rewrite, no iteration
//! ever reports convergence, and the scheduler fails the compile at its
//! iteration cap.
//!
//! That is not hypothetical. The strict-IEEE expansion of `sin` and `cos`
//! answers an out-of-domain argument with a NaN literal, and every strict-mode
//! dispatch of them failed before emission with `optimizer did not reach a
//! fixpoint after 50 iterations`.
//!
//! The relation these tests pin is bit equality: two literals are the same
//! literal exactly when they are the same bit pattern. It is the relation the
//! canonical wire fingerprint already uses, so structural equality and program
//! identity answer the same question, and it keeps `0.0` and `-0.0` apart,
//! which IEEE does not and division and copysign do.

use proptest::prelude::*;
use vyre_foundation::fp_parity::canonical_f32;
use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::{fingerprint_program, optimize};

/// A program whose whole body stores `value` to `out[0]`.
fn store_f32(value: Expr) -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), value)],
    )
}

/// The bit patterns a float literal reaches, with the classes IEEE equality
/// mishandles named rather than left to chance: quiet and payload-carrying
/// NaNs, both zeros, both infinities.
fn f32_literal() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(f32::NAN),
        Just(-f32::NAN),
        Just(f32::from_bits(0x7fc0_0001)),
        Just(f32::from_bits(0x7f80_0001)),
        Just(0.0_f32),
        Just(-0.0_f32),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(f32::MIN_POSITIVE),
        Just(1.0_f32),
        any::<u32>().prop_map(f32::from_bits),
    ]
}

proptest! {
    /// A program is equal to the clone of itself, whatever literal it holds.
    ///
    /// Every pass in the pipeline reports "changed" by negating this, so a
    /// literal that breaks it breaks convergence for every program containing
    /// one.
    #[test]
    fn a_program_holding_any_float_literal_equals_its_own_clone(value in f32_literal()) {
        let program = store_f32(Expr::f32(value));
        prop_assert_eq!(&program, &program.clone());
    }

    /// Structural equality is bit equality, and program identity is the same
    /// relation up to the one normalization the parity contract states.
    ///
    /// The two are deliberately not identical. `canonical_f32` quiets a NaN,
    /// whose payload no target preserves, and flushes a subnormal to a zero of
    /// its own sign, so a pass may rewrite a literal into a form the identity
    /// does not distinguish. What it may never do is the reverse: identity is
    /// coarser than structure, never finer, or a rewrite the fingerprint cannot
    /// see is served the other program's cached artifact.
    #[test]
    fn two_float_literals_are_one_literal_exactly_when_their_bits_match(
        left in f32_literal(),
        right in f32_literal(),
    ) {
        let same_bits = left.to_bits() == right.to_bits();
        let same_canonical = canonical_f32(left).to_bits() == canonical_f32(right).to_bits();
        let (left, right) = (store_f32(Expr::f32(left)), store_f32(Expr::f32(right)));

        prop_assert_eq!(left == right, same_bits);
        prop_assert_eq!(
            fingerprint_program(&left) == fingerprint_program(&right),
            same_canonical
        );
        prop_assert!(
            !(left == right) || fingerprint_program(&left) == fingerprint_program(&right),
            "identity must be coarser than structure, never finer"
        );
    }
}

/// The optimizer converges on a program holding a NaN literal.
///
/// The narrowest statement of the production failure: no expansion, no strict
/// mode, one literal. It fails against the pre-fix equality with
/// `optimizer did not reach a fixpoint after 50 iterations`.
#[test]
fn the_optimizer_converges_on_a_program_holding_a_nan_literal() {
    for bits in [0x7fc0_0000_u32, 0xffc0_0000, 0x7fc0_0001, 0x7f80_0001] {
        let program = store_f32(Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::f32(f32::from_bits(bits))),
            right: Box::new(Expr::load("out", Expr::u32(0))),
        });
        let optimized = optimize(program).unwrap_or_else(|error| {
            panic!("Fix: a NaN literal must not block the optimizer fixpoint: {error}")
        });
        assert!(
            optimized
                .buffers()
                .iter()
                .any(|buffer| buffer.name() == "out"),
            "Fix: the optimized program must keep the buffer it writes"
        );
    }
}
