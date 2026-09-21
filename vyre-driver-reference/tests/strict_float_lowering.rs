//! Does the parity oracle lower `FloatLoweringMode::StrictIeee` end to end?
//!
//! WHY: every backend's strict-mode answer is judged bit for bit against this
//! evaluator. An oracle that read `DispatchConfig::float_lowering` only through
//! the cache key would answer a strict request with the interpreter's own
//! native transcendental, which is the exact thing the mode denies a device, so
//! a backend that expanded correctly would be reported as diverging and a
//! backend that did not would be reported as correct.
//!
//! This crate registers no `BackendRegistration`, so it states nothing through
//! `VyreBackend::honors_float_lowering`: the oracle is reached by name, through
//! `CpuRefEvaluator`. What takes the place of that statement is the pair of
//! contracts below, which hold the evaluator to the same two outcomes a backend
//! is held to. It lowers the mode where an exact expansion exists, and it
//! refuses by name where none does.
//!
//! The operator set on both sides is derived from
//! `vyre_foundation::fp_expansion` at run time, so an approximable operator
//! that gains or loses an expansion moves between these two cases without
//! either being edited.

use vyre_driver::DispatchConfig;
use vyre_driver_reference::CpuRefEvaluator;
use vyre_foundation::fp_expansion::{
    expand_strict_transcendentals, is_strict_expandable, strict_unexpandable_unary_ops,
};
use vyre_foundation::fp_parity::{
    approximable_operations, approximable_unary_ops, FloatLoweringMode,
};
use vyre_foundation::ir::UnOp;
use vyre_test_support::strict_float_programs::f32_multiply_add_program;

/// Lanes the witness programs run over.
const LANES: u32 = 4;

/// `a`, `b` and `c` for the multiply-add witness.
///
/// Every value is positive and away from zero, because the operator sweep
/// includes `Log` and `Sqrt` and a negative argument answers NaN on both sides,
/// which compares equal and proves nothing.
fn inputs() -> [Vec<u8>; 3] {
    let a: Vec<f32> = vec![0.7, 1.3, 2.5, 3.9];
    let b: Vec<f32> = vec![1.1, 0.25, 3.5, 0.125];
    let c: Vec<f32> = vec![0.5, 2.75, 1.5, 0.0625];
    [a, b, c].map(|values| values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn strict_config() -> DispatchConfig {
    let mut config = DispatchConfig::default();
    config.float_lowering = FloatLoweringMode::StrictIeee;
    config
}

/// A strict dispatch answers exactly what the neutral expansion answers.
///
/// The oracle's strict path is `fp_expansion::expand_strict_transcendentals`
/// followed by ordinary evaluation, so the contract is that dispatching the
/// original program under the strict mode and dispatching the already-expanded
/// program under the mode that permits contraction produce the same bytes.
/// Anything else means the oracle either skipped the expansion or applied a
/// second one.
#[test]
fn a_strict_dispatch_matches_the_neutral_expansion_bit_for_bit() {
    let evaluator = CpuRefEvaluator;
    let buffers = inputs();
    let borrowed: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();

    let expandable: Vec<UnOp> = approximable_unary_ops()
        .into_iter()
        .filter(is_strict_expandable)
        .collect();
    assert!(
        !expandable.is_empty(),
        "Fix: no approximable operator has a strict expansion, so this sweep judges nothing."
    );

    let mut observably_strict = Vec::new();
    for op in expandable {
        let name = format!("{op:?}");
        let program = f32_multiply_add_program(LANES, Some(op));
        let expanded = expand_strict_transcendentals(&program)
            .unwrap_or_else(|error| {
                panic!("Fix: `{name}` has a strict expansion and must expand: {error}")
            })
            .unwrap_or_else(|| {
                panic!("Fix: a program containing `{name}` must be rewritten by the expansion")
            });
        assert!(
            approximable_operations(&expanded).is_empty(),
            "Fix: the expansion of `{name}` left an approximable operation in the program, so a \
             strict dispatch still reaches a native transcendental."
        );

        let strict = evaluator
            .evaluate(&program, &borrowed, &strict_config())
            .unwrap_or_else(|error| {
                panic!("Fix: the oracle must lower `{name}` strictly: {error}")
            });
        let neutral = evaluator
            .evaluate(&expanded, &borrowed, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!("Fix: the expanded program must evaluate under the default mode: {error}")
            });
        assert_eq!(
            strict, neutral,
            "Fix: the oracle's strict answer for `{name}` differs from the neutral expansion it \
             is defined as. Every backend's strict result is compared against this, so the two \
             have to be one computation."
        );

        let contracted = evaluator
            .evaluate(&program, &borrowed, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!("Fix: the unexpanded program must evaluate under the default mode: {error}")
            });
        if strict != contracted {
            observably_strict.push(name);
        }
    }

    assert!(
        !observably_strict.is_empty(),
        "Fix: every operator answered the same bytes under the strict mode and the mode that \
         permits contraction, so an oracle that ignored `DispatchConfig::float_lowering` \
         entirely would pass this suite."
    );
}

/// An approximable operator with no exact expansion is refused by name.
///
/// The alternative is answering a bit-identity request with the interpreter's
/// own approximate transcendental, which is a wrong answer the caller cannot
/// see. The refusal names the mode and the operator so the caller knows which
/// of the two to change.
#[test]
fn an_operation_with_no_strict_expansion_is_refused_naming_the_mode_and_the_operation() {
    let evaluator = CpuRefEvaluator;
    let buffers = inputs();
    let borrowed: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();

    let unexpandable = strict_unexpandable_unary_ops();
    assert!(
        !unexpandable.is_empty(),
        "Fix: every approximable operator now has a strict expansion. Delete this case and say \
         so, rather than leaving a sweep over an empty set."
    );

    for op in unexpandable {
        let name = format!("{op:?}");
        let program = f32_multiply_add_program(LANES, Some(op));
        let error = evaluator
            .evaluate(&program, &borrowed, &strict_config())
            .expect_err(&format!(
                "Fix: `{name}` has no exact f32 expansion, so a strict dispatch must be refused \
                 rather than answered with an approximate instruction."
            ))
            .to_string();

        assert!(
            error.contains(FloatLoweringMode::StrictIeee.cache_label()),
            "Fix: the refusal must name the mode; got `{error}`"
        );
        assert!(
            error.contains(&name),
            "Fix: the refusal must name the operation with no expansion; got `{error}`"
        );
        assert!(
            error.contains("Fix:"),
            "Fix: the refusal must carry a remediation section; got `{error}`"
        );

        evaluator
            .evaluate(&program, &borrowed, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: `{name}` is refused only under the strict mode. The mode that permits \
                     contraction must still evaluate it: {error}"
                )
            });
    }
}
