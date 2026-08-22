//! RELEASE TEST LANE 15  -  every-op random-input stress test.
//!
//! For every semantic operation with `test_inputs` and `expected_output`, generate
//! bounded random inputs (10_000 when `CI_STRESS=1`) via a manual
//! `proptest::test_runner::TestRunner`, run each through the CPU reference
//! and the wgpu backend, and assert byte-identity (int) or within-ULP
//! (float) equivalence.

#![cfg(feature = "device-tests")]
#![allow(clippy::filter_map_bool_then, clippy::unnecessary_map_or)]
#![allow(deprecated)]
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::OnceLock;
use std::time::Duration;

use proptest::test_runner::{Config, TestRunner};
use vyre::ir::Program;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::fp_parity;
use vyre_foundation::optimizer::optimize;
use vyre_libs::operation_catalog::fixture_entries;
use vyre_reference::value::Value;

mod harness;

use harness::every_op_random_inputs::{
    compare_outputs, gpu_dispatch_inputs, is_program_graph_frontier, missing_capability_reason,
    op_seed, random_amg_v_cycle_inputs, random_buffer_for, random_program_graph_frontier,
    randomize_buffer,
};

fn require_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "every_op_random_input_stress: GPU adapter probe failed. Fix: verify nvidia-smi, WGPU_BACKEND, Vulkan drivers, and wgpu adapter selection.",
        )
    })
}

/// Wall-clock ceiling on one oracle evaluation, in seconds.
///
/// The reference is the only oracle in this sweep, and a program whose loop
/// trip count is read from its own input runs for as long as that value says.
/// One random `u32` is enough to ask for four billion iterations, which turned
/// this target from a suite into a process that never returned. The ceiling is
/// far above the milliseconds every bounded op needs on the fixture extents.
/// `VYRE_ORACLE_DEADLINE_SECS` raises it, which is how a suspected offender is
/// measured rather than guessed at.
const ORACLE_DEADLINE_SECS: u64 = 20;

/// The ceiling this run enforces.
fn oracle_deadline() -> Duration {
    let seconds = std::env::var("VYRE_ORACLE_DEADLINE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(ORACLE_DEADLINE_SECS);
    Duration::from_secs(seconds)
}

/// What the oracle said about one random case.
enum Oracle {
    /// Output bytes to compare against the device.
    Answered(Vec<Vec<u8>>),
    /// Rejected or panicked, so this input has no oracle and is not compared.
    Declined,
    /// Still running at the ceiling, which is a defect in the program under
    /// test, not a property of the input.
    TimedOut,
}

/// Evaluate one case on the reference, bounded by [`oracle_deadline`].
///
/// The evaluation runs on its own thread so a trip count taken from input data
/// is reported by name instead of holding the whole target. A thread past the
/// ceiling is left to the process exit: interrupting the interpreter mid-step
/// would need a cancellation path the oracle deliberately does not have.
fn bounded_reference_eval(program: &Program, inputs: &[Value]) -> Oracle {
    let (sender, receiver) = channel();
    let program = program.clone();
    let inputs = inputs.to_vec();
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            vyre_reference::reference_eval(&program, &inputs)
        }));
        let _ = sender.send(outcome);
    });
    match receiver.recv_timeout(oracle_deadline()) {
        Ok(Ok(Ok(outputs))) => {
            Oracle::Answered(outputs.into_iter().map(|value| value.to_bytes()).collect())
        }
        Ok(Ok(Err(_)) | Err(_)) | Err(RecvTimeoutError::Disconnected) => Oracle::Declined,
        Err(RecvTimeoutError::Timeout) => Oracle::TimedOut,
    }
}

#[test]
fn every_op_random_input_stress() {
    let backend = require_backend();

    let count = if std::env::var("CI_STRESS").map_or(false, |v| v == "1") {
        10_000
    } else {
        std::env::var("VYRE_RANDOM_CASES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(8)
    };

    let mut total_cases = 0u64;
    let mut failures = Vec::new();
    let op_filter = std::env::var("VYRE_RANDOM_OP_FILTER").ok();
    let mut matched_ops = 0usize;

    for entry in fixture_entries() {
        if let Some(filter) = op_filter.as_deref() {
            if !entry.id.contains(filter) {
                continue;
            }
        }
        matched_ops += 1;

        if entry.test_inputs.is_none() || entry.expected_output.is_none() {
            panic!(
                "{} is missing test_inputs or expected_output. Fix: every op in the random-input stress sweep must provide both.",
                entry.id
            );
        }

        let program = entry.program().unwrap_or_else(|| {
            panic!(
                "Fix: stress-tested operation `{}` must provide a program",
                entry.id
            )
        });

        if let Some(reason) = missing_capability_reason(backend, &program) {
            panic!("{} missing backend capability: {reason}. Fix: wire the op or capability before stress testing.", entry.id);
        }

        let fixture_inputs = entry.test_inputs.unwrap()();
        if fixture_inputs.is_empty() {
            panic!(
                "{} has no fixture inputs. Fix: provide at least one stress seed input.",
                entry.id
            );
        }
        let fixture_case = &fixture_inputs[0];
        let buffer_lens: Vec<usize> = fixture_case.iter().map(|b| b.len()).collect();

        let seed = op_seed(entry.id);
        println!(
            "stress: {}  -  evaluating {} deterministic random cases",
            entry.id, count
        );
        let config = Config {
            rng_seed: proptest::test_runner::RngSeed::Fixed(seed),
            ..Config::default()
        };
        let mut runner = TestRunner::new(config);

        let lowered = optimize(program.clone()).expect("registered optimizer must converge");
        let mut op_cases = 0u64;
        let mut op_failures = 0usize;
        let mut op_timeouts = 0usize;

        for case_idx in 0..count {
            let mut randomized: Vec<&str> = Vec::new();
            let random_inputs = if entry.id.contains("amg_v_cycle") {
                random_amg_v_cycle_inputs(fixture_case, &mut runner)
            } else {
                let mut random_inputs = Vec::with_capacity(buffer_lens.len());
                for (buffer_idx, &len) in buffer_lens.iter().enumerate() {
                    if randomize_buffer(entry.id, &program, buffer_idx) {
                        let buffer = program
                            .buffers()
                            .get(buffer_idx)
                            .expect("fixture input index must match program buffer index");
                        let random = if is_program_graph_frontier(&program, buffer_idx) {
                            random_program_graph_frontier(&program, len, &mut runner)
                        } else {
                            random_buffer_for(entry.id, buffer, len, &mut runner)
                        };
                        randomized.push(buffer.name());
                        random_inputs.push(random);
                    } else {
                        random_inputs.push(fixture_case[buffer_idx].clone());
                    }
                }
                random_inputs
            };

            let cpu_values: Vec<Value> = random_inputs.iter().cloned().map(Value::from).collect();
            let cpu_outputs = match bounded_reference_eval(&program, &cpu_values) {
                Oracle::Answered(outputs) => outputs,
                // Reference rejected or panicked  -  no oracle for this input.
                Oracle::Declined => continue,
                Oracle::TimedOut => {
                    op_timeouts += 1;
                    failures.push(format!(
                        "{} seed={seed} case={case_idx}: the reference did not answer inside {:?} with random [{}]. Fix: bound the loop trip count by the extents of the buffer it indexes; a count read from input data spins for hours on the device as well as here.",
                        entry.id,
                        oracle_deadline(),
                        randomized.join(", ")
                    ));
                    break;
                }
            };

            let gpu_inputs = gpu_dispatch_inputs(&program, &random_inputs);
            let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.dispatch(&lowered, &gpu_inputs, &DispatchConfig::default())
            }));
            let gpu_outputs = match gpu_result {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => {
                    op_failures += 1;
                    if op_failures == 1 {
                        failures.push(format!(
                            "{} seed={} case={}: GPU dispatch error: {}",
                            entry.id, seed, case_idx, e
                        ));
                    }
                    continue;
                }
                Err(_) => {
                    op_failures += 1;
                    if op_failures == 1 {
                        failures.push(format!(
                            "{} seed={} case={}: GPU dispatch panicked",
                            entry.id, seed, case_idx
                        ));
                    }
                    continue;
                }
            };

            let tolerance = fp_parity::effective_tolerance(entry.id, &program);
            if let Err(msg) = compare_outputs(
                entry.id,
                &program,
                &cpu_outputs,
                &gpu_outputs,
                tolerance,
                seed,
            ) {
                op_failures += 1;
                if op_failures == 1 {
                    failures.push(format!("{} case={}: {msg}", entry.id, case_idx));
                }
            }

            op_cases += 1;
        }

        total_cases += op_cases;
        println!(
            "stress: {}  -  {} random cases evaluated, {} failures, {} oracle timeouts",
            entry.id, op_cases, op_failures, op_timeouts
        );
    }

    if let Some(filter) = op_filter {
        assert!(
            matched_ops > 0,
            "VYRE_RANDOM_OP_FILTER={filter:?} matched no semantic operation ids. Fix: pass a substring of the target operation id."
        );
    }

    if !failures.is_empty() {
        panic!(
            "every_op_random_input_stress failed for {} op(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    println!(
        "every_op_random_input_stress: {} total random cases passed",
        total_cases
    );
}
