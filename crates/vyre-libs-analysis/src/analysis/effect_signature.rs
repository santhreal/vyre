//! Effect-signature analysis: does a Program's effect row fit its declaration.
//!
//! A declared signature is the maximum set of effects a caller permits a Region
//! to produce. `vyre_foundation::lower::compute_program_effects` produces the
//! observed row. A Region is well-typed against signature `S` when every bit
//! set in its observed row `R` is also set in `S`.
//!
//! The row type is `vyre_foundation::lower::ProgramEffects`, the same value the
//! lowering pipeline computes and the optimizer's validate pass carries. This
//! module states no effect kinds of its own: a second kind list would drift the
//! moment a kind was added to one of them.
//!
//! An effect handler is exactly the row of effects it discharges, so composing
//! two handlers is `a | b` on the row and needs no function here.
//! [`residual_effects`] is the one operation the row type does not already name
//! for this use: what stays open after a handler runs.
//!
//! Both entry points are `const fn` over one `u32`, so neither charges
//! telemetry. They are a scalar decision, not a traversal.

use vyre_foundation::lower::ProgramEffects;

/// Verdict returned by [`check_signature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectTypeError {
    /// Effects the observed row produces that the signature does not permit.
    /// Never empty: an empty difference is reported as `Ok(())` instead.
    pub unpermitted: ProgramEffects,
}

/// Check whether `observed` fits inside `signature`.
///
/// # Errors
///
/// Returns [`EffectTypeError`] naming the effects the observed row produces
/// that the signature does not permit.
pub const fn check_signature(
    signature: ProgramEffects,
    observed: ProgramEffects,
) -> Result<(), EffectTypeError> {
    let unpermitted = observed.introduced_since(signature);
    if unpermitted.is_empty() {
        Ok(())
    } else {
        Err(EffectTypeError { unpermitted })
    }
}

/// The effects still open after a handler discharging `discharged` runs over a
/// Region producing `produced`.
///
/// Idempotent: discharging the same row twice leaves the same residual. An
/// empty `discharged` returns `produced` unchanged.
#[must_use]
#[inline]
pub const fn residual_effects(
    produced: ProgramEffects,
    discharged: ProgramEffects,
) -> ProgramEffects {
    produced.introduced_since(discharged)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every effect kind the row type declares. A residual is bit arithmetic,
    /// so it is correct only while the kinds occupy distinct single bits.
    const EVERY_KIND: [ProgramEffects; 7] = [
        ProgramEffects::BUFFER_WRITE,
        ProgramEffects::ATOMIC,
        ProgramEffects::HOST_IO,
        ProgramEffects::GPU_DISPATCH,
        ProgramEffects::BARRIER,
        ProgramEffects::ASYNC_LOAD,
        ProgramEffects::TRAP,
    ];

    #[test]
    fn every_kind_is_a_distinct_single_bit() {
        for (i, a) in EVERY_KIND.iter().enumerate() {
            assert_eq!(a.bits().count_ones(), 1, "kind {i} is not a single bit");
            for (j, b) in EVERY_KIND.iter().enumerate().skip(i + 1) {
                assert_ne!(a.bits(), b.bits(), "kinds {i} and {j} share a bit");
            }
        }
    }

    #[test]
    fn an_empty_observed_row_fits_an_empty_signature() {
        assert!(check_signature(ProgramEffects::empty(), ProgramEffects::empty()).is_ok());
    }

    #[test]
    fn a_permitted_effect_fits() {
        assert_eq!(
            check_signature(
                ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC,
                ProgramEffects::BUFFER_WRITE
            ),
            Ok(())
        );
    }

    #[test]
    fn the_verdict_names_only_the_unpermitted_effects() {
        let err = check_signature(
            ProgramEffects::BUFFER_WRITE,
            ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC,
        )
        .unwrap_err();
        assert_eq!(err.unpermitted, ProgramEffects::ATOMIC);
    }

    #[test]
    fn an_empty_signature_rejects_the_whole_observed_row() {
        let observed = ProgramEffects::HOST_IO | ProgramEffects::TRAP;
        let err = check_signature(ProgramEffects::empty(), observed).unwrap_err();
        assert_eq!(err.unpermitted, observed);
    }

    #[test]
    fn every_kind_fits_a_signature_permitting_it() {
        for kind in EVERY_KIND {
            assert!(
                check_signature(kind, kind).is_ok(),
                "{kind:?} rejects itself"
            );
        }
    }

    #[test]
    fn an_empty_handler_discharges_nothing() {
        for kind in EVERY_KIND {
            assert_eq!(residual_effects(kind, ProgramEffects::empty()), kind);
        }
    }

    #[test]
    fn a_handler_discharges_its_own_kinds_and_passes_the_rest() {
        let produced = ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC;
        let residual = residual_effects(produced, ProgramEffects::BUFFER_WRITE);
        assert_eq!(residual, ProgramEffects::ATOMIC);
    }

    #[test]
    fn discharging_twice_changes_nothing() {
        let produced = ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC;
        let once = residual_effects(produced, ProgramEffects::BUFFER_WRITE);
        assert_eq!(residual_effects(once, ProgramEffects::BUFFER_WRITE), once);
    }

    #[test]
    fn a_handler_covering_the_row_leaves_nothing_open() {
        let produced = ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC;
        assert!(residual_effects(produced, produced).is_empty());
    }

    /// Handler composition is row union, so discharging a composed handler once
    /// must equal discharging each half in turn.
    #[test]
    fn composed_discharge_equals_sequential_discharge() {
        let produced =
            ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC | ProgramEffects::HOST_IO;
        let a = ProgramEffects::BUFFER_WRITE;
        let b = ProgramEffects::ATOMIC;
        assert_eq!(
            residual_effects(produced, a | b),
            residual_effects(residual_effects(produced, a), b)
        );
    }

    /// A residual that still has open effects is exactly what the signature
    /// check rejects, so the two operations must agree.
    #[test]
    fn a_nonempty_residual_is_the_signature_verdict() {
        let produced = ProgramEffects::BUFFER_WRITE | ProgramEffects::ATOMIC | ProgramEffects::TRAP;
        for signature in EVERY_KIND {
            let residual = residual_effects(produced, signature);
            assert_eq!(
                residual.is_empty(),
                check_signature(signature, produced).is_ok(),
                "residual and verdict disagree for {signature:?}"
            );
        }
    }
}
