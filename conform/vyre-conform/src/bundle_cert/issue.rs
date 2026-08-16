//! Issuing a bundle certificate from the reference interpreter.

use vyre::ir::Program;
use vyre_conform_spec::{BundleCertificate, ConformanceCase, CERTIFICATE_SCHEMA_VERSION};
use vyre_reference::value::Value;

use crate::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};

use super::canonical::{canonicalise_corpus, hash_output_stream, hex32};
use super::error::BundleCertError;

/// Run one witness through the reference interpreter into `outputs`.
pub fn reference_dispatch<'a>(
    program: &Program,
    witness: &'a ConformanceCase,
    input_plan: &'a WitnessInputPlan,
    planned_inputs: &mut Vec<&'a [u8]>,
    values: &mut Vec<Value>,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), BundleCertError> {
    plan_witness_inputs_into(&witness.inputs, input_plan, planned_inputs).map_err(|message| {
        BundleCertError::WitnessPlanningFailed {
            witness: witness.name.clone(),
            message,
        }
    })?;
    values.clear();
    for input in planned_inputs.iter().copied() {
        values.push(Value::from(input));
    }
    let evaluated = vyre_reference::reference_eval(program, values).map_err(|e| {
        BundleCertError::ReferenceFailed {
            witness: witness.name.clone(),
            message: e.to_string(),
        }
    })?;
    outputs.clear();
    outputs.extend(evaluated.into_iter().map(|value| value.to_bytes()));
    Ok(())
}

/// Issue a fresh [`BundleCertificate`] from the CPU reference.
///
/// Runs every witness through `vyre_reference::reference_eval`, captures the
/// output stream, blake3s it, and packs the result alongside the
/// bundle wire hash. Caller supplies timestamp + signature (the cert
/// body is the input the signer sees  -  sign after issue).
///
/// # Errors
///
/// - [`BundleCertError::EmptyBundle`]  -  `program_wire_bytes` empty.
/// - [`BundleCertError::EmptyCorpus`]  -  no witnesses.
/// - [`BundleCertError::ReferenceFailed`]  -  the reference interp
///   rejected a witness.
pub fn issue_bundle_cert(
    program: &Program,
    corpus: &[ConformanceCase],
    timestamp: &str,
    signature_ed25519: &str,
    pubkey: &str,
) -> Result<BundleCertificate, BundleCertError> {
    if corpus.is_empty() {
        return Err(BundleCertError::EmptyCorpus);
    }

    let wire_bytes = program
        .to_wire()
        .map_err(|_| BundleCertError::EmptyBundle)?;
    if wire_bytes.is_empty() {
        return Err(BundleCertError::EmptyBundle);
    }
    let bundle_hash = blake3::hash(&wire_bytes);

    let (sorted_indices, corpus_hash) = canonicalise_corpus(corpus)?;
    let input_plan = WitnessInputPlan::for_program(program).map_err(|message| {
        BundleCertError::WitnessPlanningFailed {
            witness: "certificate-issue".to_string(),
            message,
        }
    })?;

    let mut witness_values = Vec::with_capacity(program.buffers().len());
    let mut witness_inputs = Vec::with_capacity(input_plan.source_count());
    let mut witness_outputs = Vec::with_capacity(program.buffers().len());
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(sorted_indices.len() as u64).to_le_bytes());
    for idx in &sorted_indices {
        let w = &corpus[*idx];
        witness_outputs.clear();
        witness_values.clear();
        witness_inputs.clear();
        reference_dispatch(
            program,
            w,
            &input_plan,
            &mut witness_inputs,
            &mut witness_values,
            &mut witness_outputs,
        )?;
        hash_output_stream(&mut hasher, &witness_outputs);
    }
    let output_hash = {
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hash.as_bytes());
        bytes
    };

    Ok(BundleCertificate {
        version: CERTIFICATE_SCHEMA_VERSION.to_string(),
        bundle_blake3: bundle_hash.to_hex().to_string(),
        corpus_blake3: hex32(&corpus_hash),
        reference_output_blake3: hex32(&output_hash),
        witness_count: sorted_indices.len() as u64,
        timestamp: timestamp.to_string(),
        signature_ed25519: signature_ed25519.to_string(),
        pubkey: pubkey.to_string(),
    })
}
