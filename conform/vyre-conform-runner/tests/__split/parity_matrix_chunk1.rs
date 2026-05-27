// Cross-backend parity matrix: registered backends, wire shapes, and buffer comparison.
// `#![forbid(unsafe_code)]` was moved to the parent `parity_matrix.rs`
// because inner attributes cannot ride an `include!`-d chunk.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use blake3::Hash;
use inventory::iter;
use vyre::backend::backend_dispatches;
use vyre::backend::registered_backends;
use vyre::ir::{DataType, Expr, ExprNode, Node, Program};
use vyre::{BackendRegistration, DispatchConfig, VyreBackend};
use vyre_conform_runner::dispatch_grid;
use vyre_conform_runner::fp_parity::{compare_output_buffers, BufferParity};
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};
use vyre_intrinsics::harness::OpEntry as IntrinsicsOpEntry;
use vyre_libs::harness::OpEntry as LibsOpEntry;
use vyre_primitives::harness::OpEntry as PrimitivesOpEntry;
use vyre_reference::value::Value;
use vyre_spec::expr_variants;

use vyre_intrinsics as _;
use vyre_libs as _;
use vyre_primitives as _;
#[cfg(feature = "gpu")]
use vyre_driver_cuda as _;
#[cfg(feature = "gpu")]
use vyre_driver_wgpu as _;

type FixtureCases = Vec<Vec<Vec<u8>>>;
type FixtureFn = fn() -> FixtureCases;

#[derive(Clone, Copy)]
struct UnifiedEntry {
    id: &'static str,
    build: fn() -> Program,
    test_inputs: Option<FixtureFn>,
    expected_output: Option<FixtureFn>,
}

#[derive(Debug)]
struct SyntheticOpaqueExpr;

impl ExprNode for SyntheticOpaqueExpr {
    fn extension_kind(&self) -> &'static str {
        "vyre.conform.synthetic.opaque"
    }

    fn debug_identity(&self) -> &str {
        "synthetic-opaque-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        true
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        [0x5a; 32]
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        vec![0x5a]
    }
}

#[derive(Debug)]
struct Divergence {
    op_id: &'static str,
    backend_a: &'static str,
    backend_b: &'static str,
    input_hash: Hash,
    output_a_hash: Hash,
    output_b_hash: Hash,
    detail: String,
}

#[derive(Default, Debug)]
struct Summary {
    ops_total: usize,
    ops_covered: usize,
    backends_linked: usize,
    backends_runnable: usize,
    divergences: Vec<Divergence>,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
enum BackendKind {
    ReferenceBackend,
    Registered(Box<dyn VyreBackend>),
}

struct BackendRunner {
    id: &'static str,
    kind: BackendKind,
}

impl BackendRunner {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        values: &mut Vec<Value>,
    ) -> Result<Vec<Vec<u8>>, String> {
        match &self.kind {
            BackendKind::ReferenceBackend => {
                values.clear();
                for bytes in inputs {
                    values.push(Value::from(bytes.as_slice()));
                }
                vyre_reference::reference_eval(program, values)
                    .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
                    .map_err(|error| format!("reference dispatch failed: {error}"))
            }
            BackendKind::Registered(_) => {
                let mut backend_inputs = Vec::new();
                let config = dispatch_grid::config_for_program(program)?;
                self.dispatch_with_plan(program, inputs, values, None, &mut backend_inputs, &config)
            }
        }
    }

    fn dispatch_with_plan<'a>(
        &self,
        program: &Program,
        inputs: &'a [Vec<u8>],
        values: &mut Vec<Value>,
        plan: Option<&'a BackendDispatchPlan>,
        backend_inputs: &mut Vec<&'a [u8]>,
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, String> {
        match &self.kind {
            BackendKind::ReferenceBackend => {
                values.clear();
                for bytes in inputs {
                    values.push(Value::from(bytes.as_slice()));
                }
                vyre_reference::reference_eval(program, values)
                    .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
                    .map_err(|error| format!("reference dispatch failed: {error}"))
            }
            BackendKind::Registered(backend) => {
                if let Some(plan) = plan {
                    backend_dispatch_inputs_with_plan_into(inputs, plan, backend_inputs)?;
                    backend
                        .dispatch_borrowed(program, &backend_inputs, &config)
                        .map_err(|error| error.to_string())
                } else {
                    let plan_storage = backend_dispatch_plan(program)?;
                    let mut local_inputs = Vec::new();
                    backend_dispatch_inputs_with_plan_into(inputs, &plan_storage, &mut local_inputs)?;
                    backend
                        .dispatch_borrowed(program, &local_inputs, &config)
                        .map_err(|error| error.to_string())
                }
            }
        }
    }
}

#[derive(Clone)]
enum BackendInputSource {
    Fixture(usize),
    Zeroed(usize),
}

struct BackendDispatchPlan {
    sources: Vec<BackendInputSource>,
    zeroed_inputs: Vec<Vec<u8>>,
    buffer_len: usize,
}

fn backend_dispatch_plan(program: &Program) -> Result<BackendDispatchPlan, String> {
    let mut sources = Vec::with_capacity(program.buffers().len());
    let mut zeroed_inputs = Vec::with_capacity(program.buffers().len());
    for (buffer_index, buffer) in program.buffers().iter().enumerate() {
        if buffer.kind() == vyre::ir::MemoryKind::Shared
            || buffer.is_output()
            || (buffer.is_pipeline_live_out()
                && matches!(buffer.access(), vyre::ir::BufferAccess::ReadWrite))
        {
            continue;
        }
        if matches!(buffer.access(), vyre::ir::BufferAccess::ReadWrite) {
            let byte_len = usize::try_from(buffer.count())
                .ok()
                .and_then(|count| count.checked_mul(buffer.element().min_bytes()))
                .ok_or_else(|| {
                    format!(
                        "buffer `{}` has no fixture and its static byte length overflows. Fix: add fixture bytes for this read-write buffer.",
                        buffer.name()
                    )
                })?;
            let zero_index = zeroed_inputs.len();
            zeroed_inputs.push(vec![0u8; byte_len]);
            sources.push(BackendInputSource::Zeroed(zero_index));
            continue;
        }
        sources.push(BackendInputSource::Fixture(buffer_index));
    }

    Ok(BackendDispatchPlan {
        sources,
        zeroed_inputs,
        buffer_len: program.buffers().len(),
    })
}

fn backend_dispatch_inputs_with_plan_into<'a>(
    fixture_inputs: &'a [Vec<u8>],
    plan: &'a BackendDispatchPlan,
    backend_inputs: &mut Vec<&'a [u8]>,
) -> Result<(), String> {
    if fixture_inputs.len() > plan.buffer_len {
        return Err(format!(
            "fixture provided {} buffer(s) but Program declares {}. Fix: fixture cases must not exceed Program::buffers order for reference parity.",
            fixture_inputs.len(),
            plan.buffer_len
        ));
    }

    backend_inputs.clear();
    for source in &plan.sources {
        match source {
            BackendInputSource::Fixture(buffer_index) => {
                if let Some(bytes) = fixture_inputs.get(*buffer_index) {
                    backend_inputs.push(bytes.as_slice());
                    continue;
                }
                return Err(
                    "fixture omitted required input buffer. Fix: every non-output read-only/uniform buffer must be present in the witness case."
                        .to_string(),
                );
            }
            BackendInputSource::Zeroed(index) => {
                if let Some(bytes) = plan.zeroed_inputs.get(*index) {
                    backend_inputs.push(bytes.as_slice());
                    continue;
                }
                return Err("internal plan mismatch: zeroed input index is invalid.".to_string());
            }
        }
    }
    Ok(())
}

// Asserts `runners.len() >= 2`, which means at least one dispatch-capable
// backend in addition to vyre-reference must be linked. If the crate is built
// without the `gpu` feature, this test must fail loudly instead of compiling
// out the parity gate.
#[test]
fn parity_matrix_across_all_registered_ops() {
    let mut summary = Summary::default();
    let runners = backend_runners(&mut summary);
    let entries = unified_entries();
    let expr_rows = expr_variant_rows(&entries);
    let filter = env::var("VYRE_PARITY_FILTER").ok();

    assert!(
        runners.len() >= 2,
        "Fix: parity_matrix requires at least one linked dispatch-capable backend in addition to vyre-reference. Link a concrete driver crate for this gate."
    );
    assert!(
        !entries.is_empty(),
        "Fix: parity matrix linked zero OpEntry registrations. Ensure vyre-libs and vyre-intrinsics are linked into this test binary."
    );
    let missing_expr_variants = expr_variants()
        .iter()
        .copied()
        .filter(|variant| !expr_rows.contains_key(variant))
        .collect::<Vec<_>>();
    assert!(
        missing_expr_variants.is_empty(),
        "Fix: parity matrix must cover every Expr variant from vyre-spec; missing rows for {}",
        missing_expr_variants.join(", ")
    );

    for entry in entries {
        if filter.as_deref().is_some_and(|needle| {
            needle
                .strip_prefix('=')
                .map_or_else(|| !entry.id.contains(needle), |exact| entry.id != exact)
        }) {
            continue;
        }
        summary.ops_total += 1;

        let test_inputs = entry.test_inputs.unwrap_or_else(|| {
            panic!(
                "{}: missing test_inputs. Fix: every registered op must provide fixture inputs.",
                entry.id
            )
        });
        let expected_output = entry.expected_output.unwrap_or_else(|| {
            panic!(
                "{}: missing expected_output. Fix: every registered op must provide fixture oracle.",
                entry.id
            )
        });

        let program = (entry.build)();
        assert_valid(entry.id, &program, &runners);
        assert_region_chain(entry.id, &program);

        let input_cases = test_inputs();
        let expected_cases = expected_output();
        assert!(
            !input_cases.is_empty(),
            "Fix: {} registered empty test_inputs. Empty witnesses are zero execution coverage.",
            entry.id,
        );
        assert!(
            !expected_cases.is_empty(),
            "Fix: {} registered empty expected_output. Empty oracles are zero execution coverage.",
            entry.id,
        );
        assert_eq!(
            input_cases.len(),
            expected_cases.len(),
            "Fix: {} test_inputs / expected_output case count mismatch ({} vs {}).",
            entry.id,
            input_cases.len(),
            expected_cases.len()
        );

        summary.ops_covered += 1;
        let input_plan = backend_dispatch_plan(&program).unwrap_or_else(|error| {
            panic!("Fix: {} backend input plan failed: {error}", entry.id);
        });
        let grid_config = dispatch_grid::config_for_program(&program).unwrap_or_else(|error| {
            panic!("Fix: {} config_for_program failed: {error}", entry.id);
        });
        let mut reference_values = Vec::with_capacity(program.buffers().len());
        let mut outputs = Vec::<(&'static str, Vec<Vec<u8>>)>::with_capacity(runners.len());
        let mut borrowed_inputs = Vec::with_capacity(input_plan.sources.len());
        for (case_index, (inputs, expected)) in
            input_cases.iter().zip(expected_cases.iter()).enumerate()
        {
            let input_hash = hash_buffers(inputs);
            let program_hash_before = hash_program(&program);
            outputs.clear();
            borrowed_inputs.clear();

            let reference_output = runners[0]
                .dispatch(&program, inputs, &mut reference_values)
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: {} case {} reference failed: {error}",
                        entry.id, case_index
                    )
                });
            let reference_hash = hash_buffers(&reference_output);
            assert_eq!(
                hash_program(&program),
                program_hash_before,
                "Fix: {} case {} mutated the Program during dispatch; region chain must remain stable post-run.",
                entry.id,
                case_index
            );
            compare_outputs(
                entry.id,
                "reference",
                "expected_output",
                input_hash,
                &reference_output,
                expected,
                &program,
                &mut summary.divergences,
            );
            outputs.push(("reference", reference_output));

            for runner in runners.iter().skip(1) {
                match catch_unwind(AssertUnwindSafe(|| {
                    runner.dispatch_with_plan(
                        &program,
                        inputs,
                        &mut reference_values,
                        Some(&input_plan),
                        &mut borrowed_inputs,
                        &grid_config,
                    )
                })) {
                    Ok(Ok(output)) => {
                        assert_eq!(
                            hash_program(&program),
                            program_hash_before,
                            "Fix: {} case {} mutated the Program during {} dispatch; region chain must remain stable post-run.",
                            entry.id,
                            case_index,
                            runner.id
                        );
                        outputs.push((runner.id, output));
                    }
                    Ok(Err(error)) => {
                        panic!(
                            "{} on {}: backend dispatch error: {}. Fix: repair backend or op before claiming parity.",
                            entry.id, runner.id, error
                        );
                    }
                    Err(payload) => {
                        summary.divergences.push(Divergence {
                            op_id: entry.id,
                            backend_a: runner.id,
                            backend_b: "reference",
                            input_hash,
                            output_a_hash: hash_buffers(&[]),
                            output_b_hash: reference_hash,
                            detail: format!("dispatch panic: {}", panic_message(payload)),
                        });
                    }
                }
            }

            for i in 0..outputs.len() {
                for j in (i + 1)..outputs.len() {
                    let (backend_a, output_a) = &outputs[i];
                    let (backend_b, output_b) = &outputs[j];
                    compare_outputs(
                        entry.id,
                        backend_a,
                        backend_b,
                        input_hash,
                        output_a,
                        output_b,
                        &program,
                        &mut summary.divergences,
                    );
                }
            }
        }
    }

    eprintln!(
        "PARITY-SUMMARY ops_total={} ops_covered={} backends_linked={} backends_runnable={} divergences={}",
        summary.ops_total,
        summary.ops_covered,
        summary.backends_linked,
        summary.backends_runnable,
        summary.divergences.len()
    );
    for variant in expr_variants() {
        if let Some(op_ids) = expr_rows.get(variant) {
            eprintln!(
                "PARITY-EXPR-COVERAGE variant={} rows={}",
                variant,
                op_ids.join(",")
            );
        }
    }

    assert!(
        summary.ops_covered == summary.ops_total,
        "parity matrix under-coverage: ops_covered={} ops_total={}. Fix: every registered op must run at least one witness case.",
        summary.ops_covered,
        summary.ops_total
    );
    assert!(
        summary.divergences.is_empty(),
        "{}",
        format_divergences(&summary.divergences)
    );
}

fn backend_runners(summary: &mut Summary) -> Vec<BackendRunner> {
    let mut registrations: Vec<&BackendRegistration> = registered_backends().iter().copied().collect();
    registrations.sort_by(|left, right| left.id.cmp(right.id));
    summary.backends_linked = registrations.len() + 1;

    let mut runners = vec![BackendRunner {
        id: "reference",
        kind: BackendKind::ReferenceBackend,
    }];

    for registration in registrations {
        if let Some(runner) = build_backend_runner(registration) {
            runners.push(runner);
        }
    }

    summary.backends_runnable = runners.len();
    runners
}

fn build_backend_runner(registration: &BackendRegistration) -> Option<BackendRunner> {
    if !backend_dispatches(registration.id) {
        return None;
    }

    match registration.acquire() {
        Ok(backend) => Some(BackendRunner {
            id: registration.id,
            kind: BackendKind::Registered(backend),
        }),
        Err(error) => panic!(
            "Fix: registered dispatch backend `{}` failed its factory probe in the strict parity matrix: {error}",
            registration.id
        ),
    }
}

fn unified_entries() -> Vec<UnifiedEntry> {
    let libs = iter::<LibsOpEntry>.into_iter().map(|entry| UnifiedEntry {
        id: entry.id,
        build: entry.build,
        test_inputs: entry.test_inputs,
        expected_output: entry.expected_output,
    });
    let intrinsics = iter::<IntrinsicsOpEntry>
        .into_iter()
        .map(|entry| UnifiedEntry {
            id: entry.id,
            build: entry.build,
            test_inputs: entry.test_inputs,
            expected_output: entry.expected_output,
        });
    let primitives = iter::<PrimitivesOpEntry>
        .into_iter()
        .map(|entry| UnifiedEntry {
            id: entry.id,
            build: entry.build,
            test_inputs: entry.test_inputs,
            expected_output: entry.expected_output,
        });

    let synthetic = synthetic_entries().into_iter();
    let mut entries = libs
        .chain(intrinsics)
        .chain(primitives)
        .chain(synthetic)
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.id.cmp(right.id));
    entries
}
