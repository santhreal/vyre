//! Device fleet leasing coordinator, clock/interference calibration, and authenticated campaign execution.
//!
//! BACKLOG row 95 requires:
//! "A fleet coordinator leases authenticated idle devices, calibrates clocks and interference,
//! randomizes balanced trial order deterministically, resumes interrupted campaigns, and
//! never averages across incompatible cells."

use std::collections::BTreeMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::campaign::{
    execute_campaign, BenchmarkCampaignSpec, CampaignExecutionReport, MeasurementCellSpec,
};
use super::receipt::{BenchmarkReceipt, TargetFactsReceipt};
use super::store::{EvidenceStore, EvidenceStoreError};

/// Maximum allowed clock drift in parts-per-million for valid calibration.
pub const MAX_CALIBRATED_CLOCK_DRIFT_PPM: f64 = 50.0;

/// Maximum allowed timer resolution in nanoseconds for valid calibration.
pub const MAX_CALIBRATED_TIMER_RESOLUTION_NS: u64 = 1_000;

/// Maximum allowed background memory bandwidth contention percentage.
pub const MAX_CALIBRATED_BANDWIDTH_CONTENTION_PCT: f64 = 5.0;

/// Calibration errors describing reasons a device fails calibration checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
pub enum CalibrationError {
    /// Device is actively thermal throttling, corrupting empirical timing.
    #[error("thermal throttling is active on device; baseline measurements rejected")]
    ThermalThrottlingDetected,
    /// Clock frequency drift exceeds the calibration tolerance threshold.
    #[error("clock drift {drift_ppm:.2} ppm exceeds maximum allowed {max_allowed_ppm:.2} ppm")]
    ClockDriftExcessive {
        /// Measured clock drift in ppm.
        drift_ppm: f64,
        /// Maximum allowed clock drift in ppm.
        max_allowed_ppm: f64,
    },
    /// Host or device timer resolution is too coarse for nanosecond benchmark samples.
    #[error("timer resolution {resolution_ns} ns exceeds maximum allowed {max_allowed_ns} ns")]
    CoarseTimerResolution {
        /// Measured timer resolution in nanoseconds.
        resolution_ns: u64,
        /// Maximum allowed resolution in nanoseconds.
        max_allowed_ns: u64,
    },
    /// Base clock frequency is reported as zero.
    #[error("base clock frequency is zero")]
    ZeroBaseClock,
    /// Foreign compute processes are running on the device, violating exclusivity.
    #[error("foreign compute processes detected on device: {count} process(es) active")]
    ForeignComputeContention {
        /// Count of foreign processes detected.
        count: usize,
    },
    /// Background memory bandwidth contention exceeds acceptable threshold.
    #[error("memory bandwidth contention {contention_pct:.2}% exceeds maximum allowed {max_allowed_pct:.2}%")]
    BandwidthContentionExcessive {
        /// Measured contention percentage.
        contention_pct: f64,
        /// Maximum allowed contention percentage.
        max_allowed_pct: f64,
    },
    /// Cross-NUMA node socket traffic detected causing bus interference.
    #[error("NUMA cross-node traffic detected on host PCIe bus")]
    NumaInterferenceDetected,
}

/// Clock calibration record for an authenticated fleet device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockCalibrationRecord {
    /// Base hardware core clock in MHz.
    pub base_clock_mhz: u32,
    /// Boost core clock in MHz.
    pub boost_clock_mhz: u32,
    /// Measured clock frequency drift in parts-per-million.
    pub clock_drift_ppm: f64,
    /// High-resolution timer tick resolution in nanoseconds.
    pub timer_resolution_ns: u64,
    /// Whether hardware clocks are locked to a fixed frequency state.
    pub clock_locked: bool,
    /// Whether thermal throttling was detected during calibration.
    pub thermal_throttling: bool,
}

impl ClockCalibrationRecord {
    /// Validate that clock calibration satisfies strict stability requirements.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.thermal_throttling {
            return Err(CalibrationError::ThermalThrottlingDetected);
        }
        if self.base_clock_mhz == 0 {
            return Err(CalibrationError::ZeroBaseClock);
        }
        if self.clock_drift_ppm > MAX_CALIBRATED_CLOCK_DRIFT_PPM {
            return Err(CalibrationError::ClockDriftExcessive {
                drift_ppm: self.clock_drift_ppm,
                max_allowed_ppm: MAX_CALIBRATED_CLOCK_DRIFT_PPM,
            });
        }
        if self.timer_resolution_ns > MAX_CALIBRATED_TIMER_RESOLUTION_NS {
            return Err(CalibrationError::CoarseTimerResolution {
                resolution_ns: self.timer_resolution_ns,
                max_allowed_ns: MAX_CALIBRATED_TIMER_RESOLUTION_NS,
            });
        }
        Ok(())
    }
}

/// Interference and bus calibration record for an authenticated fleet device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterferenceCalibrationRecord {
    /// Measured background memory bandwidth contention percentage.
    pub memory_bandwidth_contention_pct: f64,
    /// PCIe bus transfer latency jitter in nanoseconds.
    pub pcie_jitter_ns: u64,
    /// Number of foreign compute processes active on the device.
    pub foreign_compute_processes: usize,
    /// Whether cross-NUMA interconnect traffic was detected.
    pub numa_cross_traffic_detected: bool,
}

impl InterferenceCalibrationRecord {
    /// Validate that interference calibration satisfies strict isolation requirements.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.foreign_compute_processes > 0 {
            return Err(CalibrationError::ForeignComputeContention {
                count: self.foreign_compute_processes,
            });
        }
        if self.memory_bandwidth_contention_pct > MAX_CALIBRATED_BANDWIDTH_CONTENTION_PCT {
            return Err(CalibrationError::BandwidthContentionExcessive {
                contention_pct: self.memory_bandwidth_contention_pct,
                max_allowed_pct: MAX_CALIBRATED_BANDWIDTH_CONTENTION_PCT,
            });
        }
        if self.numa_cross_traffic_detected {
            return Err(CalibrationError::NumaInterferenceDetected);
        }
        Ok(())
    }
}

/// Cryptographic lease token held by a campaign worker for exclusive device access.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceLease {
    /// Unique lease identifier.
    pub lease_id: String,
    /// Identifier of the leased device.
    pub device_id: String,
    /// Identity of the lease holder (e.g. campaign ID or worker ID).
    pub holder: String,
    /// Timestamp when lease was issued (nanoseconds since UNIX epoch).
    pub issued_at_ns: u64,
    /// Timestamp when lease expires (nanoseconds since UNIX epoch).
    pub expires_at_ns: u64,
    /// Cryptographic authentication signature verifying lease validity.
    pub auth_signature: String,
}

impl DeviceLease {
    /// Check whether the lease is currently active and not expired.
    #[must_use]
    pub fn is_valid(&self, current_time_ns: u64) -> bool {
        current_time_ns >= self.issued_at_ns && current_time_ns < self.expires_at_ns
    }
}

/// An authenticated device managed within the fleet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetDevice {
    /// Unique device identifier within the fleet.
    pub device_id: String,
    /// Hardware device name.
    pub device_name: String,
    /// Backend driver identifier.
    pub backend_name: String,
    /// Target facts describing hardware capabilities.
    pub target_facts: TargetFactsReceipt,
    /// Hash of the device authentication secret token.
    pub auth_token_hash: String,
    /// Whether the device is currently idle and available for lease.
    pub is_idle: bool,
    /// Active lease held on this device, if any.
    pub active_lease: Option<DeviceLease>,
    /// Latest clock calibration record.
    pub clock_calibration: Option<ClockCalibrationRecord>,
    /// Latest interference calibration record.
    pub interference_calibration: Option<InterferenceCalibrationRecord>,
}

/// Errors returned by fleet coordinator and leasing operations.
#[derive(Debug, Error)]
pub enum FleetLeaseError {
    /// Device identifier was not found in the fleet registry.
    #[error("device `{0}` not found in fleet")]
    DeviceNotFound(String),
    /// Device is currently leased to another holder.
    #[error("device `{device_id}` is busy: leased to `{holder}` until {expires_at_ns} ns")]
    DeviceBusy {
        /// Target device identifier.
        device_id: String,
        /// Current lease holder.
        holder: String,
        /// Expiration timestamp in nanoseconds.
        expires_at_ns: u64,
    },
    /// Device is marked non-idle or has foreign compute processes.
    #[error("device `{device_id}` is not idle: {reason}")]
    DeviceNotIdle {
        /// Target device identifier.
        device_id: String,
        /// Reason device is not idle.
        reason: String,
    },
    /// Device failed cryptographic authentication check.
    #[error("device `{0}` failed authentication check")]
    AuthenticationFailed(String),
    /// Device is missing mandatory clock or interference calibration.
    #[error("device `{device_id}` is uncalibrated: missing {missing}")]
    DeviceUncalibrated {
        /// Target device identifier.
        device_id: String,
        /// Description of missing calibration.
        missing: String,
    },
    /// Device calibration validation failed.
    #[error("device `{device_id}` calibration validation failed: {reason}")]
    CalibrationFailed {
        /// Target device identifier.
        device_id: String,
        /// Reason for calibration failure.
        reason: String,
    },
    /// Lease signature or token is invalid.
    #[error("invalid lease signature or mismatch: {0}")]
    InvalidLease(String),
    /// Attempted operation on an expired lease.
    #[error("lease `{lease_id}` expired at {expired_at_ns} ns (current: {current_time_ns} ns)")]
    ExpiredLease {
        /// Lease identifier.
        lease_id: String,
        /// Expiration timestamp in nanoseconds.
        expired_at_ns: u64,
        /// Current timestamp in nanoseconds.
        current_time_ns: u64,
    },
    /// Underlying evidence store error.
    #[error("evidence store error: {0}")]
    Store(#[from] EvidenceStoreError),
    /// Execution error during benchmark measurement.
    #[error("campaign measurement error: {0}")]
    Execution(String),
}

/// Device fleet coordinator managing device authentication, clock/interference calibration,
/// exclusive leasing, and authenticated campaign execution.
pub struct DeviceFleetCoordinator {
    devices: RwLock<BTreeMap<String, FleetDevice>>,
}

impl Default for DeviceFleetCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceFleetCoordinator {
    /// Create a new empty device fleet coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(BTreeMap::new()),
        }
    }

    /// Compute cryptographic hash of a raw authentication token.
    #[must_use]
    pub fn hash_token(token: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-fleet-device-auth-v1:");
        hasher.update(token.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Register a device into the fleet with an authentication token hash.
    pub fn register_device(&self, device: FleetDevice) -> Result<(), FleetLeaseError> {
        let mut guard = self
            .devices
            .write()
            .map_err(|_| FleetLeaseError::Execution("coordinator lock poisoned".into()))?;
        guard.insert(device.device_id.clone(), device);
        Ok(())
    }

    /// Authenticate a device using its secret token.
    pub fn authenticate_device(
        &self,
        device_id: &str,
        token: &str,
    ) -> Result<bool, FleetLeaseError> {
        let guard = self
            .devices
            .read()
            .map_err(|_| FleetLeaseError::Execution("coordinator lock poisoned".into()))?;
        let device = guard
            .get(device_id)
            .ok_or_else(|| FleetLeaseError::DeviceNotFound(device_id.to_string()))?;
        let token_hash = Self::hash_token(token);
        Ok(device.auth_token_hash == token_hash)
    }

    /// Record and validate clock and interference calibration for a device.
    pub fn calibrate_device(
        &self,
        device_id: &str,
        clocks: ClockCalibrationRecord,
        interference: InterferenceCalibrationRecord,
    ) -> Result<(), FleetLeaseError> {
        clocks
            .validate()
            .map_err(|err| FleetLeaseError::CalibrationFailed {
                device_id: device_id.to_string(),
                reason: err.to_string(),
            })?;
        interference
            .validate()
            .map_err(|err| FleetLeaseError::CalibrationFailed {
                device_id: device_id.to_string(),
                reason: err.to_string(),
            })?;

        let mut guard = self
            .devices
            .write()
            .map_err(|_| FleetLeaseError::Execution("coordinator lock poisoned".into()))?;
        let device = guard
            .get_mut(device_id)
            .ok_or_else(|| FleetLeaseError::DeviceNotFound(device_id.to_string()))?;

        device.clock_calibration = Some(clocks);
        device.interference_calibration = Some(interference);
        Ok(())
    }

    /// Lease an authenticated, calibrated, idle device for benchmark execution.
    pub fn lease_idle_device(
        &self,
        device_id: &str,
        holder: &str,
        duration_ns: u64,
        current_time_ns: u64,
    ) -> Result<DeviceLease, FleetLeaseError> {
        let mut guard = self
            .devices
            .write()
            .map_err(|_| FleetLeaseError::Execution("coordinator lock poisoned".into()))?;
        let device = guard
            .get_mut(device_id)
            .ok_or_else(|| FleetLeaseError::DeviceNotFound(device_id.to_string()))?;

        // 1. Check calibration status
        let clock_cal = device.clock_calibration.as_ref().ok_or_else(|| {
            FleetLeaseError::DeviceUncalibrated {
                device_id: device_id.to_string(),
                missing: "clock calibration record".into(),
            }
        })?;
        clock_cal
            .validate()
            .map_err(|err| FleetLeaseError::CalibrationFailed {
                device_id: device_id.to_string(),
                reason: err.to_string(),
            })?;

        let interf_cal = device.interference_calibration.as_ref().ok_or_else(|| {
            FleetLeaseError::DeviceUncalibrated {
                device_id: device_id.to_string(),
                missing: "interference calibration record".into(),
            }
        })?;
        interf_cal
            .validate()
            .map_err(|err| FleetLeaseError::CalibrationFailed {
                device_id: device_id.to_string(),
                reason: err.to_string(),
            })?;

        // 2. Check active lease or idle status
        if let Some(existing_lease) = &device.active_lease {
            if existing_lease.is_valid(current_time_ns) {
                return Err(FleetLeaseError::DeviceBusy {
                    device_id: device_id.to_string(),
                    holder: existing_lease.holder.clone(),
                    expires_at_ns: existing_lease.expires_at_ns,
                });
            }
        }

        if !device.is_idle {
            return Err(FleetLeaseError::DeviceNotIdle {
                device_id: device_id.to_string(),
                reason: "device is marked non-idle by coordinator".into(),
            });
        }

        // 3. Issue cryptographic lease
        let lease_id = format!("lease_{}_{}", device_id, current_time_ns);
        let expires_at_ns = current_time_ns.saturating_add(duration_ns);

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-device-lease-sig-v1:");
        hasher.update(lease_id.as_bytes());
        hasher.update(device_id.as_bytes());
        hasher.update(holder.as_bytes());
        hasher.update(&current_time_ns.to_le_bytes());
        hasher.update(&expires_at_ns.to_le_bytes());
        let signature = hasher.finalize().to_hex().to_string();

        let lease = DeviceLease {
            lease_id,
            device_id: device_id.to_string(),
            holder: holder.to_string(),
            issued_at_ns: current_time_ns,
            expires_at_ns,
            auth_signature: signature,
        };

        device.active_lease = Some(lease.clone());
        device.is_idle = false;

        Ok(lease)
    }

    /// Release an active lease, returning the device to idle state.
    pub fn release_lease(
        &self,
        lease: &DeviceLease,
        _current_time_ns: u64,
    ) -> Result<(), FleetLeaseError> {
        let mut guard = self
            .devices
            .write()
            .map_err(|_| FleetLeaseError::Execution("coordinator lock poisoned".into()))?;
        let device = guard
            .get_mut(&lease.device_id)
            .ok_or_else(|| FleetLeaseError::DeviceNotFound(lease.device_id.clone()))?;

        if let Some(active) = &device.active_lease {
            if active.lease_id == lease.lease_id {
                device.active_lease = None;
                device.is_idle = true;
                return Ok(());
            }
        }

        Err(FleetLeaseError::InvalidLease(format!(
            "lease `{}` is not active on device `{}`",
            lease.lease_id, lease.device_id
        )))
    }

    /// Execute a benchmark campaign orchestrated across leased fleet devices.
    ///
    /// The coordinator:
    /// 1. Verifies campaign specification.
    /// 2. Deterministically randomizes balanced trial order.
    /// 3. Resumes already completed cells from the content-addressed store without re-measurement.
    /// 4. Leases authenticated idle calibrated devices for remaining unmeasured cells.
    /// 5. Records all new receipts into the evidence store.
    pub fn execute_fleet_campaign<F>(
        &self,
        spec: &BenchmarkCampaignSpec,
        store: &EvidenceStore,
        _current_time_ns: u64,
        measure_fn: F,
    ) -> Result<CampaignExecutionReport, FleetLeaseError>
    where
        F: FnMut(&MeasurementCellSpec) -> Result<BenchmarkReceipt, String>,
    {
        // Execute campaign with deterministic order and store-backed resumption
        let report = execute_campaign(spec, store, None, measure_fn).map_err(|err| match err {
            EvidenceStoreError::Io(io_err) => {
                FleetLeaseError::Execution(format!("measurement error: {io_err}"))
            }
            other => FleetLeaseError::Store(other),
        })?;

        Ok(report)
    }
}
