//! One canonical representative per algebraic-law variant, derived from the
//! closed law-family vocabulary.
//!
//! The former array was hand-written and two members behind
//! [`AlgebraicLaw`], so `CategoricalIdentity` and `CategoricalAssociative` had
//! no representative and the check that claimed to hold this list to the enum
//! compared its length to the other hand-written list instead. Both are derived
//! from [`crate::LawFamily`] now, whose mapping to `AlgebraicLaw` is an
//! exhaustive match with no catch-all.

use crate::algebraic_law::AlgebraicLaw;

/// Return one canonical representative for every [`AlgebraicLaw`] enum variant.
///
/// Parameterized variants carry stable placeholder payloads that exercise the
/// variant shape without claiming to enumerate every possible payload value.
#[must_use]
pub fn all_algebraic_laws() -> &'static [AlgebraicLaw] {
    crate::law_family::law_family_representatives()
}
