//! Contracts for `vyre_driver::shadow`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use std::sync::{Arc, Mutex};
use vyre_driver::shadow::{
    assert_exhaustive_byte_identity, ConformanceCase, ConformanceError, ConformanceMatrix,
    ReferenceExecutor,
};
use vyre_driver::{BackendError, CompiledPipeline, DispatchConfig};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

type FakeRun = dyn Fn(&[&[u8]]) -> Result<Vec<Vec<u8>>, BackendError> + Send + Sync;

struct FakePipeline {
    id: String,
    run: Arc<FakeRun>,
}

impl vyre_driver::sealed::Sealed for FakePipeline {}

impl CompiledPipeline for FakePipeline {
    fn id(&self) -> &str {
        &self.id
    }

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        (self.run)(inputs)
    }
}

fn sample_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    )
}

fn witness_matrix() -> ConformanceMatrix {
    ConformanceMatrix::new(
        u32_witnesses()
            .into_iter()
            .map(|witness| {
                ConformanceCase::new(
                    format!("u32:{witness:#010x}"),
                    vec![witness.to_le_bytes().to_vec()],
                )
            })
            .collect(),
    )
}

#[test]
fn empty_matrix_is_rejected() {
    let pipeline: Arc<dyn CompiledPipeline> = Arc::new(FakePipeline {
        id: "fake".into(),
        run: Arc::new(|inputs| Ok(inputs.iter().map(|row| row.to_vec()).collect())),
    });
    let reference = ReferenceExecutor::new(|_, inputs| Ok(inputs.to_vec()));

    let error = assert_exhaustive_byte_identity(
        pipeline.as_ref(),
        &sample_program(),
        &reference,
        &ConformanceMatrix::default(),
        &DispatchConfig::default(),
    )
    .expect_err("empty witness inventories must be rejected");

    assert!(matches!(error, ConformanceError::EmptyMatrix));
}

#[test]
fn exhaustive_matrix_passes_matching_outputs() {
    let pipeline: Arc<dyn CompiledPipeline> = Arc::new(FakePipeline {
        id: "fake".into(),
        run: Arc::new(|inputs| Ok(inputs.iter().map(|row| row.to_vec()).collect())),
    });
    let reference = ReferenceExecutor::new(|_, inputs| Ok(inputs.to_vec()));

    assert_exhaustive_byte_identity(
        pipeline.as_ref(),
        &sample_program(),
        &reference,
        &witness_matrix(),
        &DispatchConfig::default(),
    )
    .expect("Fix: matching backend/reference outputs must pass the exhaustive matrix; restore this invariant before continuing.");
}

#[test]
fn exhaustive_matrix_catches_divergence_hidden_by_sampling() {
    let hidden_witness = 0xDEAD_BEEF_u32.to_le_bytes().to_vec();
    let seen = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let seen_clone = Arc::clone(&seen);
    let pipeline: Arc<dyn CompiledPipeline> = Arc::new(FakePipeline {
        id: "fake".into(),
        run: Arc::new(move |inputs| {
            seen_clone.lock().unwrap().push(inputs[0].to_vec());
            if inputs[0] == hidden_witness.as_slice() {
                Ok(vec![0_u32.to_le_bytes().to_vec()])
            } else {
                Ok(inputs.iter().map(|row| row.to_vec()).collect())
            }
        }),
    });
    let reference = ReferenceExecutor::new(|_, inputs| Ok(inputs.to_vec()));

    let error = assert_exhaustive_byte_identity(
        pipeline.as_ref(),
        &sample_program(),
        &reference,
        &witness_matrix(),
        &DispatchConfig::default(),
    )
    .expect_err("one divergent witness must fail exhaustive conformance");

    match error {
        ConformanceError::Diverged { event, .. } => {
            assert_eq!(event.case_label, "u32:0xdeadbeef");
            assert_eq!(event.inputs, vec![0xDEAD_BEEF_u32.to_le_bytes().to_vec()]);
            assert_eq!(event.backend_output, vec![0_u32.to_le_bytes().to_vec()]);
            assert_eq!(
                event.reference_output,
                vec![0xDEAD_BEEF_u32.to_le_bytes().to_vec()]
            );
        }
        other => panic!("expected divergence event, got {other:?}"),
    }

    assert_eq!(
        seen.lock().unwrap().len(),
        u32_witnesses().len(),
        "the conformance matrix must execute every witness tuple exactly once"
    );
}

fn u32_witnesses() -> Vec<u32> {
    let mut out = vec![
        0u32,
        1,
        2,
        3,
        u32::MAX,
        u32::MAX - 1,
        0x8000_0000,
        0x7FFF_FFFF,
        0xAAAA_AAAA,
        0x5555_5555,
        0xDEAD_BEEF,
        0xCAFE_F00D,
    ];
    let mut state = 0xD5E4_A7B9_3C6D_102Fu64;
    for _ in 0..24 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push((z as u32) ^ ((z >> 32) as u32));
    }
    out
}
