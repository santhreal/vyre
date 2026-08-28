//! Admission and variant selection for one guarded artifact set.
//!
//! A runtime holding several artifacts for one graph has to pick one, and the
//! only legal way to pick is to evaluate the guards the compiler authenticated.
//! Anything else is retuning: reading a shape and choosing a launch, or
//! preferring the variant that ran fastest last time, is schedule selection
//! happening after the artifact froze it.
//!
//! So selection here is total and closed. The whole set is admitted at once, for
//! one required target payload format and one authenticated target identity.
//! Selection evaluates guards in the order the set states and returns an
//! artifact that was compiled, scored, and retained before admission. It cannot
//! alter a schedule, because it never touches one.

use std::collections::BTreeMap;

use vyre_megakernel::specialization::{
    AxisValue, PortfolioEnvelope, RemainderKind, SpecializationAxis, SpecializationContract,
    VariantGuard,
};
use vyre_megakernel::{Digest, TargetPayloadFormat};

use super::{admit_envelope, AdmittedArtifact, ArtifactAdmissionError};

/// One authenticated guarded artifact set, with every variant materializable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedPortfolio {
    contract: SpecializationContract,
    remainder_kind: RemainderKind,
    target_identity: Digest,
    order: Vec<usize>,
    variants: Vec<(VariantGuard, AdmittedArtifact)>,
    remainder: Option<AdmittedArtifact>,
}

impl AdmittedPortfolio {
    /// The contract the set was compiled under.
    #[must_use]
    pub const fn contract(&self) -> &SpecializationContract {
        &self.contract
    }

    /// What serves facts no guard admits.
    #[must_use]
    pub const fn remainder_kind(&self) -> RemainderKind {
        self.remainder_kind
    }

    /// The target identity the set was compiled for and admitted against.
    #[must_use]
    pub const fn target_identity(&self) -> Digest {
        self.target_identity
    }

    /// Admitted variants and their guards, in canonical guard order.
    #[must_use]
    pub fn variants(&self) -> &[(VariantGuard, AdmittedArtifact)] {
        &self.variants
    }

    /// The admitted generic remainder, when the set carries one.
    #[must_use]
    pub const fn remainder(&self) -> Option<&AdmittedArtifact> {
        self.remainder.as_ref()
    }

    /// The admitted artifact that serves one complete set of trusted facts.
    ///
    /// The facts are checked against the contract's declared domain first, so a
    /// workload the set never covered is refused rather than served by the
    /// remainder, which was compiled for that domain.
    ///
    /// # Errors
    ///
    /// Returns an error when the facts fall outside the declared domain, when a
    /// guard's own terms cannot hold at once, or when no guard admits the facts
    /// and the set declares its remainder unsupported.
    pub fn select(
        &self,
        facts: &BTreeMap<SpecializationAxis, AxisValue>,
    ) -> Result<&AdmittedArtifact, ArtifactAdmissionError> {
        self.contract.admits_facts(facts)?;
        for index in &self.order {
            let (guard, admitted) = &self.variants[*index];
            if guard.admits_facts(facts)? {
                return Ok(admitted);
            }
        }
        self.remainder.as_ref().ok_or_else(|| {
            ArtifactAdmissionError::from(unsupported_workload(&self.contract, facts))
        })
    }
}

/// Decode and authenticate a guarded artifact set, then admit every member for
/// one exact payload format against one authenticated target identity.
///
/// # Errors
///
/// Returns [`ArtifactAdmissionError`] when the bytes are not an authentic set,
/// when its guards no longer prove, when the authenticated target is not the one
/// the set was compiled for, or when any member lacks the required payload
/// format. A set is admitted whole: one member without the required format
/// rejects the set, because a runtime that admitted the rest would serve some
/// requests and fail others out of one authenticated product.
pub fn admit_portfolio(
    portfolio_bytes: &[u8],
    required_format: &TargetPayloadFormat,
    target_identity: Digest,
) -> Result<AdmittedPortfolio, ArtifactAdmissionError> {
    let envelope = PortfolioEnvelope::from_bytes(portfolio_bytes)?;
    envelope.require_target_identity(target_identity)?;
    let order = envelope.evaluation_order();
    let contract = envelope.contract().clone();
    let remainder_kind = envelope.remainder_kind();
    let mut variants = Vec::with_capacity(envelope.variants().len());
    for (guard, member) in envelope.variants() {
        variants.push((
            guard.clone(),
            admit_envelope(member.clone(), required_format)?,
        ));
    }
    let remainder = envelope
        .remainder()
        .map(|member| admit_envelope(member.clone(), required_format))
        .transpose()?;
    Ok(AdmittedPortfolio {
        contract,
        remainder_kind,
        target_identity,
        order,
        variants,
        remainder,
    })
}

/// The rejection a set with an unsupported remainder answers with.
fn unsupported_workload(
    contract: &SpecializationContract,
    facts: &BTreeMap<SpecializationAxis, AxisValue>,
) -> vyre_megakernel::CompileError {
    let stated = facts
        .keys()
        .filter(|axis| contract.axes().contains_key(axis))
        .map(SpecializationAxis::field)
        .collect::<Vec<_>>()
        .join(", ");
    vyre_megakernel::unsupported_workload(format!(
        "no admitted variant serves the stated facts on [{stated}] and the set declares its remainder unsupported"
    ))
}
