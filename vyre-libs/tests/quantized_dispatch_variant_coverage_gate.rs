//! Every quantized INT4 dispatch entry point validates the backend's readback
//! the same way: `decode_*_output_exact` rejects a byte count other than the
//! exact one the kernel writes, after `expect_one_output` has taken the sole
//! buffer.
//!
//! Each entry point used to pin that contract in its own suite, and the copies
//! disagreed about which half to pin. `i4x8_batched_matmul_top1_f32_scaled_via`
//! was the only one that rejected an output SHORTER than the contract; every
//! other suite rejected only an output LONGER than it. Both directions hold for
//! all six entry points, so no suite was wrong about behaviour and every suite
//! was missing half the boundary.
//!
//! The buffer count reaching readback is not a backend choice:
//! `execute_single_program` returns one buffer per written graph value, so a
//! count other than one means the entry point's program declares a writable
//! buffer count other than one. That half is pinned as the program property it
//! is, and the canonical boundary owns rejection of an executor that omits or
//! invents a graph value.
//!
//! This gate owns the contract for the whole family. Its member set is read
//! from the `pub use` re-export block of
//! `vyre_libs::solvers::quantized_dispatch` at run time, so an entry point
//! added without a row here fails rather than going uncovered.

mod bounded_compile_policy;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use vyre_foundation::ir::GraphValueId;
use vyre_libs::solvers::quantized_dispatch::{
    i4x8_batched_matmul_f32_scaled_via, i4x8_batched_matmul_top1_f32_scaled_via,
    i4x8_batched_matvec_f32_scaled_via, i4x8_dot_f32_scaled_via, i4x8_matvec_f32_scaled_via,
    unpack_i4x8_via,
};
use vyre_megakernel::{
    writable_graph_values, Digest, SemanticExecutionError, SemanticExecutionOutput,
    SemanticExecutionRequest, SemanticExecutor,
};

/// A dispatcher that returns exactly the output buffers it was handed, so the
/// entry point's readback validation is the only thing under test. The written
/// graph value count of each call is recorded, because that count decides how
/// many buffers readback sees.
struct FixedOutputDispatcher {
    outputs: Vec<Vec<u8>>,
    written: Mutex<Vec<usize>>,
}

impl FixedOutputDispatcher {
    fn new(outputs: Vec<Vec<u8>>) -> Self {
        Self {
            outputs,
            written: Mutex::new(Vec::new()),
        }
    }

    fn written_counts(&self) -> Vec<usize> {
        self.written
            .lock()
            .expect("fixture recording lock is uncontended")
            .clone()
    }
}

impl SemanticExecutor for FixedOutputDispatcher {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let node = &request.logical().graph().nodes()[0];
        // A read-write output buffer carries a retained input value, not a
        // node output port, so the written set comes from the canonical
        // derivation rather than `node.outputs`.
        let written = writable_graph_values(node);
        self.written
            .lock()
            .expect("fixture recording lock is uncontended")
            .push(written.len());
        if self.outputs.len() != written.len() {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: fixed-output executor received {} outputs for {} graph values.",
                self.outputs.len(),
                written.len()
            )));
        }
        Ok(SemanticExecutionOutput {
            artifact: Digest([1; 32]),
            payload: Digest([2; 32]),
            outputs: written.into_iter().zip(self.outputs.clone()).collect(),
        })
    }
}

/// A dispatcher that breaks the canonical output contract instead of the
/// readback contract: one graph value short, or one value the graph never
/// declared.
struct MalformedDispatcher {
    surplus: bool,
    filler: Vec<u8>,
    written: Mutex<Vec<u32>>,
}

impl MalformedDispatcher {
    fn new(surplus: bool, filler: Vec<u8>) -> Self {
        Self {
            surplus,
            filler,
            written: Mutex::new(Vec::new()),
        }
    }

    fn first_written_value(&self) -> u32 {
        *self
            .written
            .lock()
            .expect("fixture recording lock is uncontended")
            .first()
            .expect("the entry point reached the executor")
    }
}

impl SemanticExecutor for MalformedDispatcher {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let node = &request.logical().graph().nodes()[0];
        let written = writable_graph_values(node);
        self.written
            .lock()
            .expect("fixture recording lock is uncontended")
            .extend(written.iter().map(|value| value.0));
        let mut outputs = BTreeMap::new();
        if self.surplus {
            for value in written {
                outputs.insert(value, self.filler.clone());
            }
            outputs.insert(GraphValueId(u32::MAX), self.filler.clone());
        }
        Ok(SemanticExecutionOutput {
            artifact: Digest([1; 32]),
            payload: Digest([2; 32]),
            outputs,
        })
    }
}

/// One quantized dispatch entry point, the exact byte count its kernel writes
/// into the single output buffer, and a call that reaches its readback stage
/// with an otherwise valid shape.
struct EntryPoint {
    name: &'static str,
    output_bytes: usize,
    call: Box<dyn Fn(&dyn SemanticExecutor) -> Result<(), SemanticExecutionError>>,
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
                unpack_i4x8_via(dispatcher, &bounded_compile_policy::policy(), &[0], 8).map(drop)
            }),
        },
        EntryPoint {
            name: "i4x8_dot_f32_scaled_via",
            // One f32 scalar.
            output_bytes: 4,
            call: Box::new(|dispatcher| {
                i4x8_dot_f32_scaled_via(
                    dispatcher,
                    &bounded_compile_policy::policy(),
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
                    &bounded_compile_policy::policy(),
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
                    &bounded_compile_policy::policy(),
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
                    &bounded_compile_policy::policy(),
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
                    &bounded_compile_policy::policy(),
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

/// The buffer count readback sees follows from the program: one buffer per
/// written graph value. Every entry point writes exactly one, which is what
/// makes its `expect_one_output` stage pass, so a program that grew or lost a
/// writable buffer fails here.
#[test]
fn every_quantized_entry_point_writes_exactly_one_graph_value() {
    for entry in entry_points() {
        let dispatcher = FixedOutputDispatcher::new(vec![vec![0u8; entry.output_bytes]]);
        assert!(
            (entry.call)(&dispatcher).is_ok(),
            "Fix: {} must accept a single output buffer of exactly {} bytes.",
            entry.name,
            entry.output_bytes
        );
        assert_eq!(
            dispatcher.written_counts(),
            vec![1],
            "Fix: {} must declare exactly one writable buffer per dispatch; readback takes the sole buffer.",
            entry.name
        );
    }
}

/// A value count other than the declared one is rejected by the canonical
/// boundary before any entry point's readback runs, in both directions.
#[test]
fn the_boundary_rejects_an_executor_that_omits_or_invents_a_graph_value() {
    for entry in entry_points() {
        let omitting = MalformedDispatcher::new(false, vec![0u8; entry.output_bytes]);
        let Err(error) = (entry.call)(&omitting) else {
            panic!(
                "Fix: {} must reject an executor that returned no graph value.",
                entry.name
            );
        };
        assert_eq!(
            error.to_string(),
            format!(
                "semantic artifact execution failed: executor omitted canonical output value {}. Fix: return every graph output exactly once",
                omitting.first_written_value()
            ),
            "Fix: {} must surface the canonical omitted-value rejection.",
            entry.name
        );

        let inventing = MalformedDispatcher::new(true, vec![0u8; entry.output_bytes]);
        let Err(error) = (entry.call)(&inventing) else {
            panic!(
                "Fix: {} must reject an executor that returned an undeclared graph value.",
                entry.name
            );
        };
        assert_eq!(
            error.to_string(),
            "semantic artifact execution failed: executor returned 1 undeclared output value(s). Fix: return only canonical graph outputs",
            "Fix: {} must surface the canonical undeclared-value rejection.",
            entry.name
        );
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
            let dispatcher = FixedOutputDispatcher::new(vec![vec![0u8; len]]);
            let Err(error) = (entry.call)(&dispatcher) else {
                panic!("Fix: {} must reject {label}.", entry.name);
            };
            let expected = format!(
                "semantic artifact execution failed: Fix: {} expected {exact} output bytes, got {len}.",
                entry.name
            );
            assert_eq!(
                error.to_string(),
                expected,
                "Fix: {} must report the byte-count contract verbatim for {label}, with one error prefix.",
                entry.name
            );
        }
    }
}

/// Negative control for the rejection rows: the byte count each row declares is
/// the one the entry point actually accepts, so no rejection above can pass
/// because the row's constant is wrong.
#[test]
fn every_quantized_entry_point_accepts_the_exact_contracted_byte_count() {
    for entry in entry_points() {
        let dispatcher = FixedOutputDispatcher::new(vec![vec![0u8; entry.output_bytes]]);
        assert!(
            (entry.call)(&dispatcher).is_ok(),
            "Fix: {} must accept a single output buffer of exactly {} bytes; this gate's other rows assert against that count.",
            entry.name,
            entry.output_bytes
        );
    }
}
