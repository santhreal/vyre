//! Conformance coordinator: schedules content-addressed cases into disposable
//! worker processes with explicit budgets and device leases.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use vyre_conform_spec::{
    verify_receipts_for_certificate, CasePayload, CertificateRejection, DeviceLease, WorkerBudget,
    WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus,
};

use vyre_foundation::failure_domain::reclaim_poisoned_mutex;

use crate::backend_selection::backend_registration;
use crate::worker::{current_binary_digest, current_environment_digest, DEFAULT_WORKER_SECRET};

static LEASE_COUNTER: AtomicU64 = AtomicU64::new(1);
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// The subsystem every lease poison report names as the owner.
const OWNER: &str = "the conformance device lease manager";

/// A lease set records devices this process took. Discarding it leaves a lease
/// held with nothing left to release it, and refusing it forever stops the run
/// on the first panic in any worker.
const ACTIVE: &str = "the set of leases currently held";

/// Refusing this set fails open: a device the run quarantined is handed out
/// again on the next acquisition.
const QUARANTINED: &str = "the set of quarantined leases";

/// Thread-safe manager for device leases across disposable worker processes.
#[derive(Debug, Default)]
pub struct DeviceLeaseManager {
    active_leases: Mutex<HashSet<String>>,
    quarantined_leases: Mutex<HashSet<String>>,
}

impl DeviceLeaseManager {
    /// Construct a fresh lease manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_leases: Mutex::new(HashSet::new()),
            quarantined_leases: Mutex::new(HashSet::new()),
        }
    }

    /// Acquire a unique device lease for the specified backend.
    pub fn acquire_lease(&self, backend_id: &str) -> Result<DeviceLease, String> {
        let quarantined = reclaim_poisoned_mutex(&self.quarantined_leases, OWNER, QUARANTINED);

        let id_num = LEASE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let lease_id = format!("lease-{backend_id}-{id_num:06}");
        let lease_token = format!(
            "tok-{backend_id}-{id_num:06}-{}",
            current_environment_digest()
        );

        if quarantined.contains(&lease_id) {
            return Err(format!("device lease `{lease_id}` is quarantined"));
        }

        reclaim_poisoned_mutex(&self.active_leases, OWNER, ACTIVE).insert(lease_id.clone());

        Ok(DeviceLease::new(
            lease_id,
            backend_id,
            0,
            format!("{backend_id}-device-0"),
            lease_token,
        ))
    }

    /// Release a device lease after clean execution.
    pub fn release_lease(&self, lease: &DeviceLease) {
        reclaim_poisoned_mutex(&self.active_leases, OWNER, ACTIVE).remove(&lease.lease_id);
    }

    /// Quarantine a device lease after failure, leak, or driver loss.
    pub fn quarantine_lease(&self, lease: &DeviceLease, _reason: &str) {
        reclaim_poisoned_mutex(&self.active_leases, OWNER, ACTIVE).remove(&lease.lease_id);
        reclaim_poisoned_mutex(&self.quarantined_leases, OWNER, QUARANTINED)
            .insert(lease.lease_id.clone());
    }

    /// Check if a lease ID is quarantined.
    #[must_use]
    pub fn is_quarantined(&self, lease_id: &str) -> bool {
        reclaim_poisoned_mutex(&self.quarantined_leases, OWNER, QUARANTINED).contains(lease_id)
    }
}

/// Central coordinator that schedules cases into disposable worker processes.
pub struct WorkerCoordinator {
    auth_secret: Vec<u8>,
    lease_manager: DeviceLeaseManager,
}

impl Default for WorkerCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerCoordinator {
    /// Create a new coordinator with an internal authentication secret.
    #[must_use]
    pub fn new() -> Self {
        Self::with_secret(DEFAULT_WORKER_SECRET.to_vec())
    }

    /// Create a new coordinator with a custom authentication secret.
    #[must_use]
    pub fn with_secret(auth_secret: Vec<u8>) -> Self {
        Self {
            auth_secret,
            lease_manager: DeviceLeaseManager::new(),
        }
    }

    /// Reference to the device lease manager.
    #[must_use]
    pub const fn lease_manager(&self) -> &DeviceLeaseManager {
        &self.lease_manager
    }

    /// Execute a reference case in a disposable worker process.
    pub fn execute_reference(
        &self,
        case: &CasePayload,
        budget: Option<WorkerBudget>,
    ) -> WorkerReceipt {
        let budget = budget.unwrap_or_default();
        let req_id = format!(
            "req-ref-{}",
            REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let binary_digest = current_binary_digest();
        let env_digest = current_environment_digest();

        let request = WorkerRequest::new_reference(
            req_id,
            case.clone(),
            budget,
            &binary_digest,
            &binary_digest,
            &env_digest,
        );

        self.execute_request(request)
    }

    /// Execute a production case in a disposable worker process with a device lease.
    pub fn execute_production(
        &self,
        backend_id: &str,
        case: &CasePayload,
        budget: Option<WorkerBudget>,
    ) -> WorkerReceipt {
        let budget = budget.unwrap_or_default();
        let lease = match self.lease_manager.acquire_lease(backend_id) {
            Ok(l) => l,
            Err(e) => {
                let mut receipt = WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: format!("req-prod-failed-lease"),
                    case_id: case.case_id.clone(),
                    mode: WorkerMode::Production,
                    status: WorkerStatus::Quarantined {
                        reason: format!("lease acquisition failed: {e}"),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: None,
                    payload_blake3: None,
                    target_facts_blake3: None,
                    compiler_binary_blake3: current_binary_digest(),
                    runner_binary_blake3: current_binary_digest(),
                    environment_blake3: current_environment_digest(),
                    device_lease_id: None,
                    elapsed_ms: 0,
                    peak_memory_bytes: 0,
                    auth_tag: String::new(),
                };
                receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
                return receipt;
            }
        };

        let req_id = format!(
            "req-prod-{}",
            REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let binary_digest = current_binary_digest();
        let env_digest = current_environment_digest();
        let target_facts_digest = backend_registration(backend_id)
            .ok()
            .and_then(|reg| reg.acquire().ok())
            .map(|handle| {
                let facts = handle.device_profile().compile_facts();
                let debug_str = format!("{facts:?}");
                blake3::hash(debug_str.as_bytes()).to_hex().to_string()
            });

        let request = WorkerRequest::new_production(
            req_id,
            case.clone(),
            lease.clone(),
            budget,
            &binary_digest,
            &binary_digest,
            target_facts_digest,
            &env_digest,
        );

        let receipt = self.execute_request(request);
        match &receipt.status {
            WorkerStatus::Success => {
                self.lease_manager.release_lease(&lease);
            }
            WorkerStatus::Timeout { .. }
            | WorkerStatus::DriverLost { .. }
            | WorkerStatus::Leak { .. } => {
                self.lease_manager
                    .quarantine_lease(&lease, &receipt.status.to_string());
            }
            _ => {
                self.lease_manager.release_lease(&lease);
            }
        }
        receipt
    }

    /// Execute both reference and production cases in separate disposable worker processes.
    pub fn execute_case_pair(
        &self,
        backend_id: &str,
        case: &CasePayload,
        budget: Option<WorkerBudget>,
    ) -> (WorkerReceipt, WorkerReceipt) {
        let ref_receipt = self.execute_reference(case, budget);
        let prod_receipt = self.execute_production(backend_id, case, budget);
        (ref_receipt, prod_receipt)
    }

    /// Execute a case pair and verify both receipts for certificate issuance.
    ///
    /// # Errors
    ///
    /// Returns [`CertificateRejection`] naming any mismatch, timeout, panic, driver loss,
    /// or policy failure.
    pub fn execute_and_verify(
        &self,
        backend_id: &str,
        case: &CasePayload,
        budget: Option<WorkerBudget>,
    ) -> Result<WorkerReceipt, CertificateRejection> {
        let (ref_receipt, prod_receipt) = self.execute_case_pair(backend_id, case, budget);
        let expected_binary = current_binary_digest();
        let expected_env = current_environment_digest();
        let expected_target_facts = backend_registration(backend_id)
            .ok()
            .and_then(|reg| reg.acquire().ok())
            .map(|h| {
                let facts = h.device_profile().compile_facts();
                let debug_str = format!("{facts:?}");
                blake3::hash(debug_str.as_bytes()).to_hex().to_string()
            });

        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            &expected_binary,
            expected_target_facts.as_deref(),
            &expected_env,
            &case.numerical_policy,
            &self.auth_secret,
        )?;

        Ok(prod_receipt)
    }

    /// Spawn a disposable worker process for one request, enforce wall budget, and reap child.
    #[must_use]
    pub fn execute_request(&self, request: WorkerRequest) -> WorkerReceipt {
        let started = Instant::now();
        let exe = match std::env::current_exe() {
            Ok(path) => path,
            Err(e) => {
                let mut receipt = WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::ExecutionError {
                        message: format!("could not resolve current_exe: {e}"),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: None,
                    payload_blake3: None,
                    target_facts_blake3: None,
                    compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                    runner_binary_blake3: request.runner_binary_blake3.clone(),
                    environment_blake3: request.environment_blake3.clone(),
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms: 0,
                    peak_memory_bytes: 0,
                    auth_tag: String::new(),
                };
                receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
                return receipt;
            }
        };

        let mut cmd = std::process::Command::new(&exe);
        cmd.env("VYRE_CONFORM_WORKER", "1");
        cmd.env("VYRE_CONFORM_WORKER_SECRET", hex::encode(&self.auth_secret));
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let exe_str = exe.to_string_lossy();
        let is_test_binary = exe_str.contains("deps/all_tests")
            || exe_str.contains("deps/cert_regression_pin")
            || exe_str.contains("deps/all_tests_device_tests")
            || std::env::args().any(|arg| arg.contains("test") || arg == "--exact");

        if is_test_binary {
            cmd.args(["--exact", "vyre_conform_worker_entry", "--nocapture"]);
        } else {
            cmd.arg("worker");
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut receipt = WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::ExecutionError {
                        message: format!("spawn worker failed for {exe:?}: {e}"),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: None,
                    payload_blake3: None,
                    target_facts_blake3: None,
                    compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                    runner_binary_blake3: request.runner_binary_blake3.clone(),
                    environment_blake3: request.environment_blake3.clone(),
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms: 0,
                    peak_memory_bytes: 0,
                    auth_tag: String::new(),
                };
                receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
                return receipt;
            }
        };

        // Write the request to child stdin
        if let Some(mut stdin) = child.stdin.take() {
            if let Ok(req_bytes) = serde_json::to_vec(&request) {
                let _ = stdin.write_all(&req_bytes);
                let _ = stdin.write_all(b"\n");
                let _ = stdin.flush();
            }
        }

        // Bounded wall timeout poll loop
        let wall_timeout = Duration::from_millis(request.budget.wall_timeout_ms);
        let mut exit_status = None;
        while started.elapsed() < wall_timeout {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exit_status = Some(status);
                    break;
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut receipt = WorkerReceipt {
                        receipt_version: WorkerReceipt::SCHEMA_VERSION,
                        request_id: request.request_id.clone(),
                        case_id: request.case.case_id.clone(),
                        mode: request.mode,
                        status: WorkerStatus::DriverLost {
                            message: format!("try_wait failed: {e}"),
                        },
                        outputs: Vec::new(),
                        outputs_blake3: String::new(),
                        artifact_blake3: None,
                        payload_blake3: None,
                        target_facts_blake3: None,
                        compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                        runner_binary_blake3: request.runner_binary_blake3.clone(),
                        environment_blake3: request.environment_blake3.clone(),
                        device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                        elapsed_ms: u64::try_from(started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                        peak_memory_bytes: 0,
                        auth_tag: String::new(),
                    };
                    receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
                    return receipt;
                }
            }
        }

        // Handle timeout
        let status = match exit_status {
            Some(s) => s,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                let mut receipt = WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::Timeout {
                        elapsed_ms,
                        budget_ms: request.budget.wall_timeout_ms,
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: None,
                    payload_blake3: None,
                    target_facts_blake3: None,
                    compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                    runner_binary_blake3: request.runner_binary_blake3.clone(),
                    environment_blake3: request.environment_blake3.clone(),
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms,
                    peak_memory_bytes: 0,
                    auth_tag: String::new(),
                };
                receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
                return receipt;
            }
        };

        // If exit was non-zero, capture stderr and report failure
        if !status.success() {
            let mut stderr_bytes = Vec::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_end(&mut stderr_bytes);
            }
            let stderr_str = String::from_utf8_lossy(&stderr_bytes);
            let worker_status = if stderr_str.contains("panicked") {
                WorkerStatus::Panicked {
                    message: stderr_str.trim().to_string(),
                }
            } else {
                WorkerStatus::DriverLost {
                    message: format!("worker process failed with {status}: {stderr_str}"),
                }
            };
            let mut receipt = WorkerReceipt {
                receipt_version: WorkerReceipt::SCHEMA_VERSION,
                request_id: request.request_id.clone(),
                case_id: request.case.case_id.clone(),
                mode: request.mode,
                status: worker_status,
                outputs: Vec::new(),
                outputs_blake3: String::new(),
                artifact_blake3: None,
                payload_blake3: None,
                target_facts_blake3: None,
                compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                runner_binary_blake3: request.runner_binary_blake3.clone(),
                environment_blake3: request.environment_blake3.clone(),
                device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                peak_memory_bytes: 0,
                auth_tag: String::new(),
            };
            receipt.auth_tag = receipt.compute_auth_tag(&self.auth_secret);
            return receipt;
        }

        // If exit was success, read and decode stdout receipt
        let mut stdout_bytes = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout.read_to_end(&mut stdout_bytes);
        }

        let stdout_str = String::from_utf8_lossy(&stdout_bytes);
        let parsed_receipt: Option<WorkerReceipt> = stdout_str
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with('{') && trimmed.ends_with('}') {
                    serde_json::from_str::<WorkerReceipt>(trimmed).ok()
                } else {
                    None
                }
            })
            .or_else(|| serde_json::from_slice::<WorkerReceipt>(&stdout_bytes).ok());

        match parsed_receipt {
            Some(receipt) => {
                if !receipt.verify_auth_tag(&self.auth_secret) {
                    let mut prot_receipt = WorkerReceipt {
                        receipt_version: WorkerReceipt::SCHEMA_VERSION,
                        request_id: request.request_id.clone(),
                        case_id: request.case.case_id.clone(),
                        mode: request.mode,
                        status: WorkerStatus::ProtocolViolation {
                            message: "worker receipt failed authentication tag verification"
                                .to_string(),
                        },
                        outputs: Vec::new(),
                        outputs_blake3: String::new(),
                        artifact_blake3: None,
                        payload_blake3: None,
                        target_facts_blake3: None,
                        compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                        runner_binary_blake3: request.runner_binary_blake3.clone(),
                        environment_blake3: request.environment_blake3.clone(),
                        device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                        elapsed_ms: u64::try_from(started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                        peak_memory_bytes: 0,
                        auth_tag: String::new(),
                    };
                    prot_receipt.auth_tag = prot_receipt.compute_auth_tag(&self.auth_secret);
                    prot_receipt
                } else {
                    receipt
                }
            }
            None => {
                let stdout_preview = String::from_utf8_lossy(&stdout_bytes);
                let mut err_receipt = WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::ProtocolViolation {
                        message: format!(
                            "could not find valid worker receipt JSON in stdout: {stdout_preview}"
                        ),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: None,
                    payload_blake3: None,
                    target_facts_blake3: None,
                    compiler_binary_blake3: request.compiler_binary_blake3.clone(),
                    runner_binary_blake3: request.runner_binary_blake3.clone(),
                    environment_blake3: request.environment_blake3.clone(),
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    peak_memory_bytes: 0,
                    auth_tag: String::new(),
                };
                err_receipt.auth_tag = err_receipt.compute_auth_tag(&self.auth_secret);
                err_receipt
            }
        }
    }
}
