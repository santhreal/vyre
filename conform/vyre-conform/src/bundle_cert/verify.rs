//! Re-running a corpus and holding it against an issued certificate.

use vyre::ir::Program;
use vyre_conform_spec::{BundleCertificate, ConformanceCase};
use vyre_driver::BackendRegistration;
use vyre_reference::value::Value;

use crate::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use crate::ProductionSession;

use super::canonical::{canonicalise_corpus, hash_output_stream, hex32};
use super::error::BundleCertError;
use super::issue::reference_dispatch;

/// Verify a [`BundleCertificate`] through the production artifact route.
///
/// The program is compiled to a canonical artifact, target-compiled,
/// authenticated, materialized, and then submitted for every witness.
///
/// # Errors
///
/// Returns [`BundleCertError`] when certificate identity, production
/// compilation, target materialization, binding, submission, or output parity
/// fails.
pub fn verify_bundle_with_backend(
    cert: &BundleCertificate,
    program: &Program,
    backend: &'static BackendRegistration,
    corpus: &[ConformanceCase],
) -> Result<(), BundleCertError> {
    let production = ProductionSession::compile(program, backend).map_err(|error| {
        BundleCertError::ProductionFailed {
            witness: "certificate-verification".to_string(),
            message: error.to_string(),
        }
    })?;
    verify_bundle_with(
        cert,
        program,
        corpus,
        |_program, witness, input_plan, _values, borrowed_inputs, outputs| {
            plan_witness_inputs_into(&witness.inputs, input_plan, borrowed_inputs).map_err(
                |message| BundleCertError::WitnessPlanningFailed {
                    witness: witness.name.clone(),
                    message,
                },
            )?;
            let dispatched = production.submit(borrowed_inputs).map_err(|error| {
                BundleCertError::ProductionFailed {
                    witness: witness.name.clone(),
                    message: error.to_string(),
                }
            })?;
            outputs.clear();
            outputs.extend(dispatched);
            Ok(())
        },
    )
}

/// Verify a [`BundleCertificate`] against the CPU reference. Useful
/// at issue time for self-checks and in CI when a GPU isn't present.
/// The reference-only verifier is guaranteed to match the cert when
/// the cert was issued from the same `(program, corpus)`; treat a
/// failure here as a bug in the hashing, not a correctness failure.
///
/// # Errors
///
/// Same surface as [`verify_bundle_with_backend`], except every
/// divergence reports "reference" rather than a backend id.
pub fn verify_bundle_against_reference(
    cert: &BundleCertificate,
    program: &Program,
    corpus: &[ConformanceCase],
) -> Result<(), BundleCertError> {
    verify_bundle_with(
        cert,
        program,
        corpus,
        |p, w, input_plan, values, borrowed_inputs, outputs| {
            reference_dispatch(p, w, input_plan, borrowed_inputs, values, outputs)
        },
    )
}

fn verify_bundle_with<F>(
    cert: &BundleCertificate,
    program: &Program,
    corpus: &[ConformanceCase],
    mut dispatch: F,
) -> Result<(), BundleCertError>
where
    F: for<'a> FnMut(
        &Program,
        &'a ConformanceCase,
        &'a WitnessInputPlan,
        &mut Vec<Value>,
        &mut Vec<&'a [u8]>,
        &mut Vec<Vec<u8>>,
    ) -> Result<(), BundleCertError>,
{
    cert.validate_schema_version()
        .map_err(|error| BundleCertError::UnsupportedSchemaVersion(error.found().to_string()))?;
    if corpus.is_empty() {
        return Err(BundleCertError::EmptyCorpus);
    }

    let wire_bytes = program
        .to_wire()
        .map_err(|_| BundleCertError::EmptyBundle)?;
    let observed_bundle = blake3::hash(&wire_bytes).to_hex().to_string();
    if observed_bundle != cert.bundle_blake3 {
        return Err(BundleCertError::BundleHashMismatch {
            expected: cert.bundle_blake3.clone(),
            observed: observed_bundle,
        });
    }

    let (sorted_indices, corpus_hash) = canonicalise_corpus(corpus)?;
    let observed_corpus = hex32(&corpus_hash);
    if observed_corpus != cert.corpus_blake3 {
        return Err(BundleCertError::CorpusHashMismatch {
            expected: cert.corpus_blake3.clone(),
            observed: observed_corpus,
        });
    }

    // CRITIQUE_CONFORM_2026-04-23 L1: witness_count was declared
    // but never validated. Reject mismatches here so a tampered
    // cert that claims a bogus count is surfaced with both values
    // named instead of silently accepted.
    let observed_count = sorted_indices.len() as u64;
    if observed_count != cert.witness_count {
        return Err(BundleCertError::WitnessCountMismatch {
            expected: cert.witness_count,
            observed: observed_count,
        });
    }

    let input_plan = WitnessInputPlan::for_program(program).map_err(|message| {
        BundleCertError::WitnessPlanningFailed {
            witness: "certificate-verification".to_string(),
            message,
        }
    })?;
    let mut values = Vec::with_capacity(program.buffers().len());
    let mut outputs = Vec::with_capacity(program.buffers().len());
    let mut borrowed_inputs = Vec::with_capacity(input_plan.source_count());
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(sorted_indices.len() as u64).to_le_bytes());
    for idx in &sorted_indices {
        let w = &corpus[*idx];
        values.clear();
        borrowed_inputs.clear();
        outputs.clear();
        dispatch(
            program,
            w,
            &input_plan,
            &mut values,
            &mut borrowed_inputs,
            &mut outputs,
        )?;
        hash_output_stream(&mut hasher, &outputs);
    }
    let observed_output = {
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hash.as_bytes());
        hex32(&bytes)
    };
    if observed_output != cert.reference_output_blake3 {
        return Err(BundleCertError::OutputHashMismatch {
            expected: cert.reference_output_blake3.clone(),
            observed: observed_output,
        });
    }

    Ok(())
}
