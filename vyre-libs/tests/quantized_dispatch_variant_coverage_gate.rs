//! Every quantized INT4 dispatch entry point validates the backend's readback
//! the same way, in two stages: `expect_one_output` rejects a buffer count
//! other than one, then `decode_*_output_exact` rejects a byte count other
//! than the exact one the kernel writes.
//!
//! Each entry point used to pin that contract in its own suite, and the copies
//! disagreed about which half to pin. `i4x8_batched_matmul_top1_f32_scaled_via`
//! was the only one that rejected two buffers, and the only one that rejected
//! an output SHORTER than the contract; every other suite rejected only an
//! output LONGER than it, and none rejected two buffers. Both halves hold for
//! all six entry points, so no suite was wrong about behaviour and every suite
//! was missing half the boundary.
//!
//! This gate owns the contract for the whole family. Its member set is read
//! from the `pub use` re-export block of
//! `vyre_libs::solvers::quantized_dispatch` at run time, so an entry point
//! added without a row here fails rather than going uncovered.

mod semantic_execution_support;

use std::collections::BTreeSet;

use vyre_libs::solvers::quantized_dispatch::{
    i4x8_batched_matmul_f32_scaled_via, i4x8_batched_matmul_top1_f32_scaled_via,
    i4x8_batched_matvec_f32_scaled_via, i4x8_dot_f32_scaled_via, i4x8_matvec_f32_scaled_via,
    unpack_i4x8_via,
};
use vyre_megakernel::{
    Digest, SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest,
    SemanticExecutor,
};

/// A dispatcher that returns exactly the output buffers it was handed, so the
/// entry point's readback validation is the only thing under test.
struct FixedOutputDispatcher {
    outputs: Vec<Vec<u8>>,
}

impl SemanticExecutor for FixedOutputDispatcher {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let node = &request.logical().graph().nodes()[0];
        if self.outputs.len() != node.outputs.len() {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: fixed-output executor received {} outputs for {} graph values.",
                self.outputs.len(),
                node.outputs.len()
            )));
        }
        Ok(SemanticExecutionOutput {
            artifact: Digest([1; 32]),
            payload: Digest([2; 32]),
            outputs: node
                .outputs
                .iter()
                .copied()
                .zip(self.outputs.clone())
                .collect(),
        })
    }
}

/// One quantized dispatch entry point, the exact byte count its kernel writes
/// into the single output buffer, and a call that reaches its readback stage
/// with an otherwise valid shape.
struct EntryPoint {
    name: &'static str,
    output_bytes: usize,
    call: Box<dyn Fn(&FixedOutputDispatcher) -> Result<(), SemanticExecutionError>>,
}

/// `cols = 8` packs into exactly one u32 word per row, so every shape below is
/// one word per weight row and one word per activation batch.
fn entry_points() -> Vec<EntryPoint> {
    vec![
        EntryPoint {
            name: "unpack_i4x8_via",
            // Eight i32 lanes.
            output_bytes: 32,
            call: Box::new(|dispatcher| {
                unpack_i4x8_via(dispatcher, &semantic_execution_support::policy(), &[0], 8)
                    .map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_dot_f32_scaled_via",
            // One f32 scalar.
            output_bytes: 4,
            call: Box::new(|dispatcher| {
                i4x8_dot_f32_scaled_via(
                    dispatcher,
                    &semantic_execution_support::policy(),
                    &[0],
                    &[0],
                    0.5,
                    0.25,
                    8,
                )
                .map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_matvec_f32_scaled_via",
            // rows = 1.
            output_bytes: 4,
            call: Box::new(|dispatcher| {
                i4x8_matvec_f32_scaled_via(
                    dispatcher,
                    &semantic_execution_support::policy(),
                    &[0],
                    &[0.0; 8],
                    &[0.5],
                    1,
                    8,
                )
                .map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_batched_matvec_f32_scaled_via",
            // batch * rows = 2.
            output_bytes: 8,
            call: Box::new(|dispatcher| {
                i4x8_batched_matvec_f32_scaled_via(
                    dispatcher,
                    &semantic_execution_support::policy(),
                    &[0],
                    &[0.0; 16],
                    &[0.5],
                    2,
                    1,
                    8,
                )
                .map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_batched_matmul_f32_scaled_via",
            // batch * rows = 2.
            output_bytes: 8,
            call: Box::new(|dispatcher| {
                i4x8_batched_matmul_f32_scaled_via(
                    dispatcher,
                    &semantic_execution_support::policy(),
                    &[0],
                    &[0, 0],
                    &[0.5],
                    &[0.25, 0.375],
                    2,
                    1,
                    8,
                )
                .map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_batched_matmul_top1_f32_scaled_via",
            // One interleaved buffer of batch * 2 f32: scores then indices.
            output_bytes: 16,
            call: Box::new(|dispatcher| {
                i4x8_batched_matmul_top1_f32_scaled_via(
                    dispatcher,
                    &semantic_execution_support::policy(),
                    &[0],
                    &[0, 0],
                    &[0.5],
                    &[0.25, 0.375],
                    2,
                    1,
                    8,
                )
                .map(drop)
            }),
        },
    ]
}

/// Names re-exported as dispatch entry points by
/// `vyre-libs/src/solvers/quantized_dispatch/mod.rs`, read from that file so a new
/// entry point enters this gate's member set without anyone editing the gate.
///
/// The `_with_scratch_into` forms share the entry point's readback code and are
/// reached through it, so only the owning `_via` name is a member.
fn published_entry_point_names() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-libs/src/solvers/quantized_dispatch/mod.rs");
    let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("Fix: quantized dispatch module root must be readable at {path:?}: {error}")
    });

    let mut names = BTreeSet::new();
    let mut in_re_export = false;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("pub use ") {
            in_re_export = true;
        }
        if in_re_export {
            for token in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if token.ends_with("_via") && token.len() > "_via".len() {
                    names.insert(token.to_string());
                }
            }
            if line.contains(';') {
                in_re_export = false;
            }
        }
    }

    assert!(
        !names.is_empty(),
        "Fix: quantized dispatch entry-point scan found no `pub use ..._via` re-export in {path:?}; the gate cannot derive its member set."
    );
    names
}

#[test]
fn every_published_quantized_entry_point_has_a_readback_contract_row() {
    let published = published_entry_point_names();
    let covered = entry_points()
        .iter()
        .map(|entry| entry.name.to_string())
        .collect::<BTreeSet<_>>();

    let missing = published.difference(&covered).collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "Fix: quantized dispatch entry points {missing:?} are published by vyre_libs::solvers::quantized_dispatch but have no row in this gate. Add a row asserting the backend-readback contract for each."
    );

    let stale = covered.difference(&published).collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "Fix: this gate has rows for {stale:?}, which vyre_libs::solvers::quantized_dispatch no longer publishes. Delete the stale rows."
    );
}

#[test]
fn every_quantized_entry_point_rejects_a_buffer_count_other_than_one() {
    for entry in entry_points() {
        let filler = vec![0u8; entry.output_bytes];

        for (label, outputs) in [
            ("zero output buffers", Vec::new()),
            ("two output buffers", vec![filler.clone(), filler.clone()]),
        ] {
            let count = outputs.len();
            let dispatcher = FixedOutputDispatcher { outputs };
            let Err(error) = (entry.call)(&dispatcher) else {
                panic!(
                    "Fix: {} must reject {label} from the backend before decoding.",
                    entry.name
                );
            };
            let expected = format!(
                "dispatcher backend error: Fix: {} expected exactly one output buffer, got {count}.",
                entry.name
            );
            assert_eq!(
                error.to_string(),
                expected,
                "Fix: {} must report the buffer-count contract verbatim for {label}.",
                entry.name
            );
        }
    }
}

#[test]
fn every_quantized_entry_point_rejects_a_byte_count_other_than_the_exact_one() {
    for entry in entry_points() {
        let exact = entry.output_bytes;

        for (label, len) in [
            ("an output shorter than the contract", exact - 4),
            ("an output longer than the contract", exact + 4),
        ] {
            let dispatcher = FixedOutputDispatcher {
                outputs: vec![vec![0u8; len]],
            };
            let Err(error) = (entry.call)(&dispatcher) else {
                panic!("Fix: {} must reject {label}.", entry.name);
            };
            let expected = format!(
                "dispatcher backend error: Fix: {} expected {exact} output bytes, got {len}.",
                entry.name
            );
            assert_eq!(
                error.to_string(),
                expected,
                "Fix: {} must report the byte-count contract verbatim for {label}.",
                entry.name
            );
        }
    }
}

/// Negative control for the two rejection tests: the byte count each row
/// declares is the one the entry point actually accepts, so neither rejection
/// above can pass because the row's constant is wrong.
#[test]
fn every_quantized_entry_point_accepts_the_exact_contracted_byte_count() {
    for entry in entry_points() {
        let dispatcher = FixedOutputDispatcher {
            outputs: vec![vec![0u8; entry.output_bytes]],
        };
        assert!(
            (entry.call)(&dispatcher).is_ok(),
            "Fix: {} must accept a single output buffer of exactly {} bytes; this gate's other rows assert against that count.",
            entry.name,
            entry.output_bytes
        );
    }
}
