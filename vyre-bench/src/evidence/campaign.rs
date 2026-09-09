//! Resumable benchmark campaign protocol, cell identity, and incompatible cell refusal.
//!
//! BACKLOG row 95 requires:
//! 1. A campaign must be resumable: an interrupted run continues rather than restarting.
//! 2. A recorded cell is never averaged with an incompatible cell (refused by name).
//! 3. Trial order is randomized deterministically from the campaign identity.
//! 4. Content addressing ensures a changed input invalidates exactly the dependent cells.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::receipt::{
    BenchmarkBudgets, BenchmarkObjective, BenchmarkReceipt, BinaryIdentityReceipt,
    EnvironmentReceipt, ResourceIdentityReceipt, SemanticGraphIdentity, TargetFactsReceipt,
    UncertaintyModelReceipt, WorkloadAndInputIdentity,
};
use super::store::{EvidenceStore, EvidenceStoreError};

/// Specification of a single benchmark measurement cell.
///
/// A cell represents a single point in the multi-dimensional parameter space
/// of workloads, graphs, compiler binaries, hardware target facts, and budgets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasurementCellSpec {
    /// Identifier for this measurement cell.
    pub cell_id: String,
    /// Case identifier.
    pub case_id: String,
    /// Workload and input identity.
    pub workload_and_input: WorkloadAndInputIdentity,
    /// Semantic graph identity.
    pub semantic_graph: SemanticGraphIdentity,
    /// Resource allocation requirements.
    pub resource_identity: ResourceIdentityReceipt,
    /// Compiler and backend driver binary identity.
    pub compiler_and_backend_binaries: BinaryIdentityReceipt,
    /// Optimization objective.
    pub objective: BenchmarkObjective,
    /// Execution and compilation budgets.
    pub budgets: BenchmarkBudgets,
    /// Target device hardware facts.
    pub target_facts: TargetFactsReceipt,
    /// Host execution environment provenance.
    pub environment: EnvironmentReceipt,
    /// Repeat sample count or trial index.
    pub repeat_count: usize,
}

impl MeasurementCellSpec {
    /// Compute the cryptographic cell identity key from input parameters.
    ///
    /// The cell key identifies the exact input conditions under which a measurement
    /// is taken. Any change to an input parameter changes this key.
    #[must_use]
    pub fn cell_identity_key(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-measurement-cell-v1:");
        let canonical = serde_json::json!({
            "workload_and_input": &self.workload_and_input,
            "semantic_graph": &self.semantic_graph,
            "resource_identity": &self.resource_identity,
            "compiler_and_backend_binaries": &self.compiler_and_backend_binaries,
            "objective": &self.objective,
            "budgets": &self.budgets,
            "target_facts": &self.target_facts,
            "environment": &self.environment,
        });
        let bytes = serde_json::to_vec(&canonical)
            .expect("Fix: keep MeasurementCellSpec fields serializable");
        hasher.update(&bytes);
        hasher.finalize().to_hex().to_string()
    }
}

/// Specification of a full benchmark campaign containing multiple measurement cells.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkCampaignSpec {
    /// Identifier for the campaign.
    pub campaign_id: String,
    /// Deterministic seed for trial ordering.
    pub seed: u64,
    /// The planned measurement cells.
    pub cells: Vec<MeasurementCellSpec>,
}

impl BenchmarkCampaignSpec {
    /// Compute the cryptographic campaign identity hash.
    #[must_use]
    pub fn campaign_identity(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-benchmark-campaign-v1:");
        hasher.update(self.campaign_id.as_bytes());
        hasher.update(&self.seed.to_le_bytes());
        for cell in &self.cells {
            hasher.update(cell.cell_identity_key().as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Generate a deterministic, balanced trial execution order across all cells.
    ///
    /// Trial order is randomized deterministically from the campaign identity to
    /// prevent systematic thermal or warmup bias while maintaining full reproducibility.
    #[must_use]
    pub fn deterministic_trial_order(&self) -> Vec<usize> {
        let n = self.cells.len();
        if n == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..n).collect();

        // Seed SplitMix64 from the first 8 bytes of the campaign identity hash
        let campaign_hash = self.campaign_identity();
        let hash_bytes = campaign_hash.as_bytes();
        let mut seed = self.seed;
        for chunk in hash_bytes.chunks(8) {
            let mut buf = [0u8; 8];
            let len = chunk.len().min(8);
            buf[..len].copy_from_slice(&chunk[..len]);
            seed = seed.wrapping_add(u64::from_le_bytes(buf));
        }

        // SplitMix64 PRNG
        let mut splitmix = move || -> u64 {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = (splitmix() as usize) % (i + 1);
            order.swap(i, j);
        }

        order
    }
}

/// Execution status of a single campaign cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CampaignCellExecutionStatus {
    /// Cell was already present in the content-addressed evidence store and was reused.
    Cached {
        /// Cell identifier.
        cell_id: String,
        /// Cell input identity key.
        cell_identity_key: String,
        /// Content address of the cached receipt.
        content_address: String,
    },
    /// Cell was executed during this campaign run and recorded into the store.
    Executed {
        /// Cell identifier.
        cell_id: String,
        /// Cell input identity key.
        cell_identity_key: String,
        /// Content address of the newly recorded receipt.
        content_address: String,
    },
}

/// Execution report for a benchmark campaign run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignExecutionReport {
    /// Identifier of the campaign.
    pub campaign_id: String,
    /// Cryptographic identity hash of the campaign.
    pub campaign_identity: String,
    /// Total number of cells in the campaign spec.
    pub total_cells: usize,
    /// Number of cells loaded from store without re-measurement.
    pub cached_cells: usize,
    /// Number of cells newly executed during this run.
    pub executed_cells: usize,
    /// The deterministic execution order that was applied.
    pub trial_execution_order: Vec<usize>,
    /// Per-cell execution statuses in execution order.
    pub cell_statuses: Vec<CampaignCellExecutionStatus>,
    /// Content addresses of all completed receipts in execution order.
    pub receipt_addresses: Vec<String>,
}

/// Execute a benchmark campaign against a content-addressed evidence store.
///
/// Resumability: If a cell's input identity key is already present in the evidence
/// store, the existing receipt is reused without re-measuring. If interrupted after
/// a certain number of cells, resuming the campaign will skip already-completed cells.
pub fn execute_campaign<F>(
    spec: &BenchmarkCampaignSpec,
    store: &EvidenceStore,
    interrupt_after: Option<usize>,
    mut measure_cell_fn: F,
) -> Result<CampaignExecutionReport, EvidenceStoreError>
where
    F: FnMut(&MeasurementCellSpec) -> Result<BenchmarkReceipt, String>,
{
    let order = spec.deterministic_trial_order();
    let mut cell_statuses = Vec::new();
    let mut receipt_addresses = Vec::new();
    let mut cached_count = 0;
    let mut executed_count = 0;

    for (processed_idx, &cell_idx) in order.iter().enumerate() {
        if let Some(limit) = interrupt_after {
            if processed_idx >= limit {
                break;
            }
        }

        let cell = &spec.cells[cell_idx];
        let cell_key = cell.cell_identity_key();

        if let Some(existing_receipt) = store.find_by_cell_key(&cell_key)? {
            let address = existing_receipt.content_address();
            cell_statuses.push(CampaignCellExecutionStatus::Cached {
                cell_id: cell.cell_id.clone(),
                cell_identity_key: cell_key,
                content_address: address.clone(),
            });
            receipt_addresses.push(address);
            cached_count += 1;
        } else {
            let receipt = measure_cell_fn(cell).map_err(|err| {
                EvidenceStoreError::Io(std::io::Error::new(std::io::ErrorKind::Other, err))
            })?;
            let address = store.put(&receipt)?;
            cell_statuses.push(CampaignCellExecutionStatus::Executed {
                cell_id: cell.cell_id.clone(),
                cell_identity_key: cell_key,
                content_address: address.clone(),
            });
            receipt_addresses.push(address);
            executed_count += 1;
        }
    }

    Ok(CampaignExecutionReport {
        campaign_id: spec.campaign_id.clone(),
        campaign_identity: spec.campaign_identity(),
        total_cells: spec.cells.len(),
        cached_cells: cached_count,
        executed_cells: executed_count,
        trial_execution_order: order,
        cell_statuses,
        receipt_addresses,
    })
}

/// Structured refusal when attempting to combine or average incompatible benchmark cells.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[error("refused: incompatible cell identity for case `{case_id}` on dimension `{dimension}`: expected `{expected}`, actual `{actual}`")]
pub struct CellIncompatibilityRefusal {
    /// Benchmark case identifier.
    pub case_id: String,
    /// The identity dimension that differed between the cells.
    pub dimension: String,
    /// Expected value from the primary/baseline cell.
    pub expected: String,
    /// Incompatible actual value from the comparator cell.
    pub actual: String,
}

/// Assert that two benchmark receipts share identical input identities and execution environments.
///
/// Refuses by name if any identity dimension differs, preventing invalid cross-device,
/// cross-binary, cross-input, or cross-graph averaging.
pub fn assert_cells_compatible(
    a: &BenchmarkReceipt,
    b: &BenchmarkReceipt,
) -> Result<(), CellIncompatibilityRefusal> {
    let case_id = &a.workload_and_input.workload_id;

    if a.workload_and_input.workload_id != b.workload_and_input.workload_id {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "workload_and_input.workload_id".into(),
            expected: a.workload_and_input.workload_id.clone(),
            actual: b.workload_and_input.workload_id.clone(),
        });
    }

    if a.workload_and_input.input_fingerprint != b.workload_and_input.input_fingerprint {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "workload_and_input.input_fingerprint".into(),
            expected: a.workload_and_input.input_fingerprint.clone(),
            actual: b.workload_and_input.input_fingerprint.clone(),
        });
    }

    if a.workload_and_input.data_shape != b.workload_and_input.data_shape {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "workload_and_input.data_shape".into(),
            expected: format!("{:?}", a.workload_and_input.data_shape),
            actual: format!("{:?}", b.workload_and_input.data_shape),
        });
    }

    if a.workload_and_input.element_count != b.workload_and_input.element_count {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "workload_and_input.element_count".into(),
            expected: a.workload_and_input.element_count.to_string(),
            actual: b.workload_and_input.element_count.to_string(),
        });
    }

    if a.semantic_graph.graph_digest != b.semantic_graph.graph_digest {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "semantic_graph.graph_digest".into(),
            expected: a.semantic_graph.graph_digest.clone(),
            actual: b.semantic_graph.graph_digest.clone(),
        });
    }

    if a.compiler_and_backend_binaries.compiler_git_commit
        != b.compiler_and_backend_binaries.compiler_git_commit
    {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "compiler_and_backend_binaries.compiler_git_commit".into(),
            expected: a.compiler_and_backend_binaries.compiler_git_commit.clone(),
            actual: b.compiler_and_backend_binaries.compiler_git_commit.clone(),
        });
    }

    if a.compiler_and_backend_binaries.backend_name != b.compiler_and_backend_binaries.backend_name
    {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "compiler_and_backend_binaries.backend_name".into(),
            expected: a.compiler_and_backend_binaries.backend_name.clone(),
            actual: b.compiler_and_backend_binaries.backend_name.clone(),
        });
    }

    if a.compiler_and_backend_binaries.driver_version
        != b.compiler_and_backend_binaries.driver_version
    {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "compiler_and_backend_binaries.driver_version".into(),
            expected: a.compiler_and_backend_binaries.driver_version.clone(),
            actual: b.compiler_and_backend_binaries.driver_version.clone(),
        });
    }

    if a.target_facts.device_name != b.target_facts.device_name {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "target_facts.device_name".into(),
            expected: a.target_facts.device_name.clone(),
            actual: b.target_facts.device_name.clone(),
        });
    }

    if a.target_facts.architecture != b.target_facts.architecture {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "target_facts.architecture".into(),
            expected: a.target_facts.architecture.clone(),
            actual: b.target_facts.architecture.clone(),
        });
    }

    if a.target_facts.compute_units != b.target_facts.compute_units {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "target_facts.compute_units".into(),
            expected: a.target_facts.compute_units.to_string(),
            actual: b.target_facts.compute_units.to_string(),
        });
    }

    if a.target_facts.subgroup_size != b.target_facts.subgroup_size {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "target_facts.subgroup_size".into(),
            expected: a.target_facts.subgroup_size.to_string(),
            actual: b.target_facts.subgroup_size.to_string(),
        });
    }

    if a.budgets.execution_deadline_ns != b.budgets.execution_deadline_ns {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "budgets.execution_deadline_ns".into(),
            expected: a.budgets.execution_deadline_ns.to_string(),
            actual: b.budgets.execution_deadline_ns.to_string(),
        });
    }

    if a.budgets.max_device_memory_bytes != b.budgets.max_device_memory_bytes {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "budgets.max_device_memory_bytes".into(),
            expected: a.budgets.max_device_memory_bytes.to_string(),
            actual: b.budgets.max_device_memory_bytes.to_string(),
        });
    }

    if a.environment.os != b.environment.os {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "environment.os".into(),
            expected: a.environment.os.clone(),
            actual: b.environment.os.clone(),
        });
    }

    if a.resource_identity.resident_bytes != b.resource_identity.resident_bytes {
        return Err(CellIncompatibilityRefusal {
            case_id: case_id.clone(),
            dimension: "resource_identity.resident_bytes".into(),
            expected: a.resource_identity.resident_bytes.to_string(),
            actual: b.resource_identity.resident_bytes.to_string(),
        });
    }

    Ok(())
}

/// Average multiple compatible benchmark cell receipts into an aggregated receipt.
///
/// Refuses to average if any pair of receipts fails cell compatibility validation.
pub fn average_compatible_cells(
    receipts: &[BenchmarkReceipt],
) -> Result<BenchmarkReceipt, CellIncompatibilityRefusal> {
    if receipts.is_empty() {
        return Err(CellIncompatibilityRefusal {
            case_id: "empty".into(),
            dimension: "count".into(),
            expected: ">= 1".into(),
            actual: "0".into(),
        });
    }

    let primary = &receipts[0];
    for other in &receipts[1..] {
        assert_cells_compatible(primary, other)?;
    }

    let mut all_samples = Vec::new();
    for r in receipts {
        all_samples.extend_from_slice(&r.raw_samples);
    }
    all_samples.sort_unstable();

    let n = all_samples.len();
    let median = if n > 0 {
        all_samples[n / 2]
    } else {
        primary.uncertainty_model.median_ns
    };

    let mean = if n > 0 {
        all_samples.iter().sum::<u64>() as f64 / n as f64
    } else {
        primary.uncertainty_model.mean_ns
    };

    let variance = if n > 1 {
        all_samples
            .iter()
            .map(|&s| {
                let diff = s as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let stddev = variance.sqrt();

    let mad = if n > 0 {
        let mut diffs: Vec<f64> = all_samples
            .iter()
            .map(|&s| (s as f64 - median as f64).abs())
            .collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        diffs[n / 2]
    } else {
        0.0
    };

    let p95 = if n > 0 {
        all_samples[((n as f64 * 0.95).floor() as usize).min(n - 1)]
    } else {
        median
    };

    let p99 = if n > 0 {
        all_samples[((n as f64 * 0.99).floor() as usize).min(n - 1)]
    } else {
        median
    };

    let std_err = if n > 1 {
        stddev / (n as f64).sqrt()
    } else {
        0.0
    };
    let conf_lower = (mean - 1.96 * std_err).max(0.0);
    let conf_upper = mean + 1.96 * std_err;

    let mut merged = primary.clone();
    merged.raw_samples = all_samples;
    merged.uncertainty_model = UncertaintyModelReceipt {
        mean_ns: mean,
        median_ns: median,
        stddev_ns: stddev,
        mad_ns: mad,
        p95_ns: p95,
        p99_ns: p99,
        confidence_95_lower_ns: conf_lower,
        confidence_95_upper_ns: conf_upper,
    };

    Ok(merged)
}
