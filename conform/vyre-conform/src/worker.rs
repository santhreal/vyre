//! Disposable conformance worker process execution.
//!
//! A worker runs in its own process, holds one device lease, decodes inputs independently,
//! executes under explicit budgets, and emits an authenticated execution receipt.

use std::io::{Read, Write};
use std::time::Instant;

use vyre::ir::Program;
use vyre_conform_spec::{hash_outputs, WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus};

use crate::backend_selection::backend_registration;
use crate::oracle::OracleSession;
use crate::production::{ProductionSession, CONFORMANCE_SCHEDULES};

/// Secret used to authenticate worker reports when none is provided via environment.
pub const DEFAULT_WORKER_SECRET: &[u8] = b"vyre.conform.worker.internal.secret.v1";

/// Run the worker stdio loop: read request from stdin, execute, write receipt to stdout.
pub fn run_worker_stdio() {
    let mut stdin_bytes = Vec::new();
    if let Err(e) = std::io::stdin().read_to_end(&mut stdin_bytes) {
        eprintln!("worker failed to read request from stdin: {e}");
        std::process::exit(1);
    }
    let secret = std::env::var("VYRE_CONFORM_WORKER_SECRET")
        .ok()
        .and_then(|hex_str| hex::decode(hex_str).ok())
        .unwrap_or_else(|| DEFAULT_WORKER_SECRET.to_vec());

    let request: WorkerRequest = match serde_json::from_slice(&stdin_bytes) {
        Ok(req) => req,
        Err(e) => {
            let error_receipt = WorkerReceipt {
                receipt_version: WorkerReceipt::SCHEMA_VERSION,
                request_id: "unknown".to_string(),
                case_id: "unknown".to_string(),
                mode: WorkerMode::Reference,
                status: WorkerStatus::ProtocolViolation {
                    message: format!("invalid JSON worker request: {e}"),
                },
                outputs: Vec::new(),
                outputs_blake3: String::new(),
                artifact_blake3: None,
                payload_blake3: None,
                target_facts_blake3: None,
                compiler_binary_blake3: String::new(),
                runner_binary_blake3: String::new(),
                environment_blake3: String::new(),
                device_lease_id: None,
                elapsed_ms: 0,
                peak_memory_bytes: 0,
                auth_tag: String::new(),
            };
            let mut signed = error_receipt;
            signed.auth_tag = signed.compute_auth_tag(&secret);
            if let Ok(json) = serde_json::to_string(&signed) {
                println!("{json}");
            }
            std::process::exit(0);
        }
    };

    let receipt = execute_worker_request(&request, &secret);
    if let Ok(json) = serde_json::to_string(&receipt) {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(json.as_bytes());
        let _ = stdout.write_all(b"\n");
        let _ = stdout.flush();
    }
}

/// Check if the worker environment variable is set; if so, run worker stdio and exit immediately.
pub fn run_worker_from_env_or_exit() {
    if std::env::var("VYRE_CONFORM_WORKER").is_ok() {
        run_worker_stdio();
        std::process::exit(0);
    }
}

/// Execute a worker request within this process and return an authenticated receipt.
#[must_use]
pub fn execute_worker_request(request: &WorkerRequest, secret: &[u8]) -> WorkerReceipt {
    let started = Instant::now();
    let compiler_binary = current_binary_digest();
    let runner_binary = current_binary_digest();
    let environment = current_environment_digest();

    let execution_result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request.mode {
            WorkerMode::Reference => execute_reference(request),
            WorkerMode::Production => execute_production(request),
        }));

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let peak_memory_bytes = estimate_process_memory();

    let mut receipt = match execution_result {
        Err(panic_payload) => {
            let msg = crate::panic_payload::panic_message(panic_payload);
            WorkerReceipt {
                receipt_version: WorkerReceipt::SCHEMA_VERSION,
                request_id: request.request_id.clone(),
                case_id: request.case.case_id.clone(),
                mode: request.mode,
                status: WorkerStatus::Panicked {
                    message: msg.to_string(),
                },
                outputs: Vec::new(),
                outputs_blake3: String::new(),
                artifact_blake3: None,
                payload_blake3: None,
                target_facts_blake3: None,
                compiler_binary_blake3: compiler_binary,
                runner_binary_blake3: runner_binary,
                environment_blake3: environment,
                device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                elapsed_ms,
                peak_memory_bytes,
                auth_tag: String::new(),
            }
        }
        Ok(Err(exec_err)) => WorkerReceipt {
            receipt_version: WorkerReceipt::SCHEMA_VERSION,
            request_id: request.request_id.clone(),
            case_id: request.case.case_id.clone(),
            mode: request.mode,
            status: WorkerStatus::ExecutionError { message: exec_err },
            outputs: Vec::new(),
            outputs_blake3: String::new(),
            artifact_blake3: None,
            payload_blake3: None,
            target_facts_blake3: None,
            compiler_binary_blake3: compiler_binary,
            runner_binary_blake3: runner_binary,
            environment_blake3: environment,
            device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
            elapsed_ms,
            peak_memory_bytes,
            auth_tag: String::new(),
        },
        Ok(Ok(success_data)) => {
            let total_output_bytes =
                success_data.outputs.iter().map(Vec::len).sum::<usize>() as u64;
            if total_output_bytes > request.budget.max_output_bytes {
                WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::Leak {
                        bytes_leaked: total_output_bytes,
                        message: format!(
                            "total output size {total_output_bytes} exceeded ceiling {}",
                            request.budget.max_output_bytes
                        ),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: success_data.artifact_blake3,
                    payload_blake3: success_data.payload_blake3,
                    target_facts_blake3: success_data.target_facts_blake3,
                    compiler_binary_blake3: compiler_binary,
                    runner_binary_blake3: runner_binary,
                    environment_blake3: environment,
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms,
                    peak_memory_bytes,
                    auth_tag: String::new(),
                }
            } else {
                let outputs_blake3 = hash_outputs(&success_data.outputs);
                WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::Success,
                    outputs: success_data.outputs,
                    outputs_blake3,
                    artifact_blake3: success_data.artifact_blake3,
                    payload_blake3: success_data.payload_blake3,
                    target_facts_blake3: success_data.target_facts_blake3,
                    compiler_binary_blake3: compiler_binary,
                    runner_binary_blake3: runner_binary,
                    environment_blake3: environment,
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms,
                    peak_memory_bytes,
                    auth_tag: String::new(),
                }
            }
        }
    };

    receipt.auth_tag = receipt.compute_auth_tag(secret);
    receipt
}

struct WorkerSuccess {
    outputs: Vec<Vec<u8>>,
    artifact_blake3: Option<String>,
    payload_blake3: Option<String>,
    target_facts_blake3: Option<String>,
}

fn execute_reference(request: &WorkerRequest) -> Result<WorkerSuccess, String> {
    let program = Program::from_wire(&request.case.program_wire)
        .map_err(|e| format!("program wire decode failed in reference worker: {e}"))?;
    let session = OracleSession::new(program);
    let inputs_borrowed: Vec<&[u8]> = request.case.inputs.iter().map(Vec::as_slice).collect();
    let outputs = session
        .execute(&inputs_borrowed)
        .map_err(|e| format!("oracle execution failed: {e}"))?;
    Ok(WorkerSuccess {
        outputs,
        artifact_blake3: None,
        payload_blake3: None,
        target_facts_blake3: None,
    })
}

fn execute_production(request: &WorkerRequest) -> Result<WorkerSuccess, String> {
    let program = Program::from_wire(&request.case.program_wire)
        .map_err(|e| format!("program wire decode failed in production worker: {e}"))?;
    let lease = request
        .device_lease
        .as_ref()
        .ok_or_else(|| "missing device lease in production worker request".to_string())?;

    let registration = backend_registration(&lease.backend_id)
        .map_err(|e| format!("backend `{}` registration failed: {e}", lease.backend_id))?;

    let mut session = ProductionSession::from_registration(&program, registration)
        .map_err(|e| format!("production session creation failed: {e}"))?;

    if let Some(ref schedule_name) = request.case.schedule_family {
        if let Some((_, required)) = CONFORMANCE_SCHEDULES
            .iter()
            .find(|(name, _)| *name == schedule_name.as_str())
        {
            session = session.requiring_schedule(*required);
        }
    }

    let inputs_borrowed: Vec<&[u8]> = request.case.inputs.iter().map(Vec::as_slice).collect();
    let execution = session
        .submit(&inputs_borrowed)
        .map_err(|e| format!("production submit failed: {e}"))?;

    let target_facts_blake3 = registration.acquire().ok().map(|handle| {
        let facts = handle.device_profile().compile_facts();
        let debug_str = format!("{facts:?}");
        blake3::hash(debug_str.as_bytes()).to_hex().to_string()
    });
    Ok(WorkerSuccess {
        outputs: execution.outputs,
        artifact_blake3: Some(execution.artifact.to_hex().to_string()),
        payload_blake3: Some(execution.payload.to_hex().to_string()),
        target_facts_blake3,
    })
}

/// Compute a canonical digest of the current binary.
#[must_use]
pub fn current_binary_digest() -> String {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Ok(bytes) = std::fs::read(&exe_path) {
            return blake3::hash(&bytes).to_hex().to_string();
        }
    }
    let fallback = format!("vyre-conform-bin-{}", env!("CARGO_PKG_VERSION"));
    blake3::hash(fallback.as_bytes()).to_hex().to_string()
}

/// Compute a canonical digest of the execution environment.
#[must_use]
pub fn current_environment_digest() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre.conform.environment.v1");
    hasher.update(std::env::consts::OS.as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    hasher.update(std::env::consts::FAMILY.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn estimate_process_memory() -> u64 {
    // Basic heuristic / /proc/self/statm reader on linux if available
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(first) = statm.split_whitespace().next() {
                if let Ok(pages) = first.parse::<u64>() {
                    return pages * 4096;
                }
            }
        }
    }
    1024 * 1024
}
