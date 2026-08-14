//! Replay capsule construction for a diverging pair: buffer hashes, hex dumps, and the
//! first mismatch classification.

use crate::operation_selection::PreparedEntry;
use vyre_conform_spec::{
    ReplayCapsule, ReplayMinimization, ReplayMismatch, REPLAY_CAPSULE_SCHEMA_VERSION,
};

pub(crate) fn build_replay_capsule(
    backend_id: &str,
    prepared: &PreparedEntry,
    case_index: usize,
    inputs: &[Vec<u8>],
    backend_outputs: &[Vec<u8>],
    reference_outputs: &[Vec<u8>],
) -> ReplayCapsule {
    ReplayCapsule {
        schema_version: REPLAY_CAPSULE_SCHEMA_VERSION,
        op_id: prepared.id.to_string(),
        backend_id: backend_id.to_string(),
        case_index,
        replay_command: format!(
            "vyre-conform dispatch --backend {} --ops {}",
            backend_id, prepared.id
        ),
        program_blake3: hex::encode(prepared.program.content_hash()),
        witness_input_blake3: hash_buffer_stream(
            b"vyre-conform/replay-capsule/witness-input/v1",
            inputs,
        ),
        reference_output_blake3: hash_buffer_stream(
            b"vyre-conform/replay-capsule/reference-output/v1",
            reference_outputs,
        ),
        backend_output_blake3: hash_buffer_stream(
            b"vyre-conform/replay-capsule/backend-output/v1",
            backend_outputs,
        ),
        witness_input_buffers_hex: hex_buffers(inputs),
        reference_output_buffers_hex: hex_buffers(reference_outputs),
        backend_output_buffers_hex: hex_buffers(backend_outputs),
        witness_input_count: inputs.len(),
        reference_output_count: reference_outputs.len(),
        backend_output_count: backend_outputs.len(),
        first_mismatch: first_replay_mismatch(backend_outputs, reference_outputs),
        minimization: ReplayMinimization {
            strategy: "single_witness_case".to_string(),
            original_case_count: prepared.cases.len(),
            retained_case_count: 1,
        },
    }
}

fn hash_buffer_stream(domain: &[u8], buffers: &[Vec<u8>]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    update_hash_len(&mut hasher, buffers.len());
    for buffer in buffers {
        update_hash_len(&mut hasher, buffer.len());
        hasher.update(buffer);
    }
    hasher.finalize().to_hex().to_string()
}

fn update_hash_len(hasher: &mut blake3::Hasher, len: usize) {
    let encoded = u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes();
    hasher.update(&encoded);
}

fn hex_buffers(buffers: &[Vec<u8>]) -> Vec<String> {
    buffers.iter().map(hex::encode).collect()
}

fn first_replay_mismatch(
    backend_outputs: &[Vec<u8>],
    reference_outputs: &[Vec<u8>],
) -> ReplayMismatch {
    if backend_outputs.len() != reference_outputs.len() {
        return ReplayMismatch {
            kind: "output_count".to_string(),
            output_index: None,
            byte_index: None,
            reference_len: Some(reference_outputs.len()),
            backend_len: Some(backend_outputs.len()),
            reference_byte: None,
            backend_byte: None,
        };
    }

    for (output_index, (backend, reference)) in backend_outputs
        .iter()
        .zip(reference_outputs.iter())
        .enumerate()
    {
        if backend.len() != reference.len() {
            return ReplayMismatch {
                kind: "output_length".to_string(),
                output_index: Some(output_index),
                byte_index: None,
                reference_len: Some(reference.len()),
                backend_len: Some(backend.len()),
                reference_byte: None,
                backend_byte: None,
            };
        }
        if let Some((byte_index, (backend_byte, reference_byte))) = backend
            .iter()
            .copied()
            .zip(reference.iter().copied())
            .enumerate()
            .find(|(_, (backend_byte, reference_byte))| backend_byte != reference_byte)
        {
            return ReplayMismatch {
                kind: "byte".to_string(),
                output_index: Some(output_index),
                byte_index: Some(byte_index),
                reference_len: Some(reference.len()),
                backend_len: Some(backend.len()),
                reference_byte: Some(reference_byte),
                backend_byte: Some(backend_byte),
            };
        }
    }

    ReplayMismatch {
        kind: "unclassified".to_string(),
        output_index: None,
        byte_index: None,
        reference_len: None,
        backend_len: None,
        reference_byte: None,
        backend_byte: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness_fixtures::backend_dispatch_plan;
    use vyre::ir::{BufferAccess, BufferDecl, DataType, Node, Program};
    use vyre_conform::dispatch_grid;
    use vyre_conform_spec::ConformanceResult;

    #[test]
    fn replay_capsule_records_hashes_and_first_byte_mismatch() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
                BufferDecl::output("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            Vec::<Node>::new(),
        );
        let input = 7u32.to_le_bytes().to_vec();
        let reference_output = vec![1, 2, 3, 4];
        let backend_output = vec![1, 2, 9, 4];
        let prepared = PreparedEntry {
            id: "test.replay_capsule",
            dispatch_config: dispatch_grid::config_for_program(&program)
                .expect("Fix: one-workgroup program must have a dispatch grid."),
            input_plan: backend_dispatch_plan(&program)
                .expect("Fix: replay-capsule test program must plan backend inputs."),
            program,
            cases: vec![vec![input.clone()]],
            reference_cases: vec![vec![reference_output.clone()]],
            convergence_max_iterations: None,
        };

        let capsule = build_replay_capsule(
            "metal",
            &prepared,
            0,
            &prepared.cases[0],
            &[backend_output],
            &[reference_output],
        );

        assert_eq!(capsule.schema_version, REPLAY_CAPSULE_SCHEMA_VERSION);
        assert_eq!(capsule.op_id, "test.replay_capsule");
        assert_eq!(capsule.backend_id, "metal");
        assert_eq!(capsule.case_index, 0);
        assert_eq!(
            capsule.replay_command,
            "vyre-conform dispatch --backend metal --ops test.replay_capsule"
        );
        assert_hex64(&capsule.program_blake3);
        assert_hex64(&capsule.witness_input_blake3);
        assert_hex64(&capsule.reference_output_blake3);
        assert_hex64(&capsule.backend_output_blake3);
        assert_eq!(capsule.witness_input_buffers_hex, vec![hex::encode(input)]);
        assert_eq!(capsule.reference_output_buffers_hex, vec!["01020304"]);
        assert_eq!(capsule.backend_output_buffers_hex, vec!["01020904"]);
        assert_eq!(capsule.witness_input_count, 1);
        assert_eq!(capsule.reference_output_count, 1);
        assert_eq!(capsule.backend_output_count, 1);
        assert_eq!(capsule.first_mismatch.kind, "byte");
        assert_eq!(capsule.first_mismatch.output_index, Some(0));
        assert_eq!(capsule.first_mismatch.byte_index, Some(2));
        assert_eq!(capsule.first_mismatch.reference_byte, Some(3));
        assert_eq!(capsule.first_mismatch.backend_byte, Some(9));
        assert_eq!(capsule.minimization.strategy, "single_witness_case");
        assert_eq!(capsule.minimization.original_case_count, 1);
        assert_eq!(capsule.minimization.retained_case_count, 1);
    }

    #[test]
    fn pair_result_omits_capsule_on_success_and_serializes_capsule_on_failure() {
        let success = ConformanceResult {
            op_id: "test.success".into(),
            backend_id: "metal".to_string(),
            passed: true,
            message: "ok".to_string(),
            replay_capsule: None,
        };
        let success_json =
            serde_json::to_value(&success).expect("Fix: success pair must serialize.");
        assert!(
            success_json.get("replay_capsule").is_none(),
            "Fix: passing pairs must not grow empty replay_capsule fields."
        );

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            Vec::<Node>::new(),
        );
        let prepared = PreparedEntry {
            id: "test.failure",
            dispatch_config: dispatch_grid::config_for_program(&program)
                .expect("Fix: output-only program must have a dispatch grid."),
            input_plan: backend_dispatch_plan(&program)
                .expect("Fix: output-only program must still have a backend plan."),
            program,
            cases: vec![vec![]],
            reference_cases: vec![vec![vec![0]]],
            convergence_max_iterations: None,
        };
        let failure = ConformanceResult {
            op_id: "test.failure".into(),
            backend_id: "metal".to_string(),
            passed: false,
            message: "diverged".to_string(),
            replay_capsule: Some(build_replay_capsule(
                "metal",
                &prepared,
                0,
                &prepared.cases[0],
                &[vec![1]],
                &[vec![0]],
            )),
        };
        let failure_json =
            serde_json::to_value(&failure).expect("Fix: failure pair must serialize.");
        let capsule = failure_json
            .get("replay_capsule")
            .expect("Fix: diverging pairs must serialize replay_capsule.");
        assert_eq!(capsule["first_mismatch"]["kind"], "byte");
        assert_eq!(capsule["first_mismatch"]["byte_index"], 0);
        assert_eq!(capsule["backend_output_buffers_hex"][0], "01");
        assert_eq!(capsule["reference_output_buffers_hex"][0], "00");
    }

    fn assert_hex64(value: &str) {
        assert_eq!(
            value.len(),
            64,
            "Fix: replay fingerprints must be BLAKE3 hex."
        );
        assert!(
            value.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Fix: replay fingerprints must contain only hex characters."
        );
    }
}
