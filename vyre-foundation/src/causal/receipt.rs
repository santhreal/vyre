//! Versioned causal receipt, critical-path reconstruction, and serialization.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::event::{CausalEvent, CounterfactualDecision};
use super::id::{CausalSpanId, TraceId};

/// Canonical schema version for CausalReceipt.
pub const CAUSAL_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Causal analysis and serialization error.
#[derive(Debug, Error)]
pub enum CausalError {
    /// Deserialized receipt has an unsupported or stale schema version.
    #[error("stale causal receipt schema version: expected {expected}, found {found}")]
    StaleSchemaVersion {
        /// Supported version.
        expected: u32,
        /// Stale version found.
        found: u32,
    },
    /// Invalid receipt structure.
    #[error("invalid receipt structure: {0}")]
    InvalidReceiptStructure(String),
    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(String),
    /// TOML serialization or deserialization failed.
    #[error("TOML error: {0}")]
    Toml(String),
    /// Span id not found in receipt.
    #[error("span id {0} not found in receipt")]
    SpanNotFound(CausalSpanId),
    /// Cycle detected in causal DAG.
    #[error("cycle detected in causal span dependencies")]
    CycleDetected,
}

/// Single receipt capturing an end-to-end causal trace across compiler and runtime stages.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CausalReceipt {
    /// Schema version for wire compatibility and fail-closed validation.
    pub schema_version: u32,
    /// Global trace identifier.
    pub trace_id: TraceId,
    /// Root span identifier.
    pub root_span_id: CausalSpanId,
    /// All recorded events in emission order.
    pub events: Vec<CausalEvent>,
    /// Reconstructed critical-path span identifiers.
    pub critical_path: Vec<CausalSpanId>,
    /// Summary of all counterfactual decisions made during compilation/dispatch.
    pub counterfactual_summary: Vec<CounterfactualDecision>,
    /// Total wall-clock duration in nanoseconds.
    pub total_wall_time_ns: u64,
    /// Total device execution duration in nanoseconds.
    pub total_device_time_ns: u64,
    /// Total heap allocations across all stages.
    pub total_allocations: usize,
    /// Total retained bytes at completion.
    pub total_retained_bytes: usize,
}

impl CausalReceipt {
    /// Create a new receipt initialized with schema version 1.
    pub fn new(trace_id: TraceId, root_span_id: CausalSpanId) -> Self {
        Self {
            schema_version: CAUSAL_RECEIPT_SCHEMA_VERSION,
            trace_id,
            root_span_id,
            events: Vec::new(),
            critical_path: Vec::new(),
            counterfactual_summary: Vec::new(),
            total_wall_time_ns: 0,
            total_device_time_ns: 0,
            total_allocations: 0,
            total_retained_bytes: 0,
        }
    }

    /// Add an event to the receipt and update cumulative metrics.
    pub fn record_event(&mut self, event: CausalEvent) {
        self.total_wall_time_ns += event.wall_time_ns;
        if let Some(dev_ns) = event.device_time_ns {
            self.total_device_time_ns += dev_ns;
        }
        self.total_allocations += event.allocations;
        self.total_retained_bytes += event.retained_bytes;
        if let Some(decision) = &event.counterfactual_decision {
            self.counterfactual_summary.push(decision.clone());
        }
        self.events.push(event);
    }

    /// Reconstruct the critical path through the causal event DAG based on wall time.
    pub fn reconstruct_critical_path(&mut self) -> Result<&[CausalSpanId], CausalError> {
        if self.events.is_empty() {
            self.critical_path.clear();
            return Ok(&self.critical_path);
        }

        let mut span_by_id = BTreeMap::new();
        let mut children_map: BTreeMap<CausalSpanId, Vec<CausalSpanId>> = BTreeMap::new();

        for event in &self.events {
            span_by_id.insert(event.span_id, event);
            if let Some(parent) = event.parent_id {
                children_map.entry(parent).or_default().push(event.span_id);
            }
        }

        // Find root or events without parent in map
        let roots: Vec<CausalSpanId> = self
            .events
            .iter()
            .filter(|e| e.parent_id.is_none() || e.span_id == self.root_span_id)
            .map(|e| e.span_id)
            .collect();

        let mut memo: BTreeMap<CausalSpanId, (u64, Vec<CausalSpanId>)> = BTreeMap::new();
        let mut visiting = BTreeSet::new();

        fn dfs(
            node: CausalSpanId,
            span_by_id: &BTreeMap<CausalSpanId, &CausalEvent>,
            children_map: &BTreeMap<CausalSpanId, Vec<CausalSpanId>>,
            memo: &mut BTreeMap<CausalSpanId, (u64, Vec<CausalSpanId>)>,
            visiting: &mut BTreeSet<CausalSpanId>,
        ) -> Result<(u64, Vec<CausalSpanId>), CausalError> {
            if let Some(cached) = memo.get(&node) {
                return Ok(cached.clone());
            }
            if visiting.contains(&node) {
                return Err(CausalError::CycleDetected);
            }
            visiting.insert(node);

            let event = span_by_id.get(&node).copied().ok_or(CausalError::SpanNotFound(node))?;
            let mut max_child_cost = 0;
            let mut best_child_path = Vec::new();

            if let Some(children) = children_map.get(&node) {
                for &child in children {
                    let (child_cost, child_path) = dfs(child, span_by_id, children_map, memo, visiting)?;
                    if child_cost >= max_child_cost {
                        max_child_cost = child_cost;
                        best_child_path = child_path;
                    }
                }
            }

            visiting.remove(&node);
            let total_cost = event.wall_time_ns + max_child_cost;
            let mut full_path = vec![node];
            full_path.extend(best_child_path);

            memo.insert(node, (total_cost, full_path.clone()));
            Ok((total_cost, full_path))
        }

        let mut longest_path = Vec::new();
        let mut max_cost = 0;

        for root in roots {
            if let Ok((cost, path)) = dfs(root, &span_by_id, &children_map, &mut memo, &mut visiting) {
                if cost >= max_cost {
                    max_cost = cost;
                    longest_path = path;
                }
            }
        }

        self.critical_path = longest_path;
        Ok(&self.critical_path)
    }

    /// Retrieve decision rationale for a given span.
    pub fn explain_decision(&self, span_id: CausalSpanId) -> Option<&CounterfactualDecision> {
        self.events
            .iter()
            .find(|e| e.span_id == span_id)
            .and_then(|e| e.counterfactual_decision.as_ref())
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, CausalError> {
        serde_json::to_string_pretty(self).map_err(|e| CausalError::Json(e.to_string()))
    }

    /// Deserialize from JSON, validating schema version fail-closed.
    pub fn from_json(json_str: &str) -> Result<Self, CausalError> {
        let receipt: Self = serde_json::from_str(json_str).map_err(|e| CausalError::Json(e.to_string()))?;
        if receipt.schema_version != CAUSAL_RECEIPT_SCHEMA_VERSION {
            return Err(CausalError::StaleSchemaVersion {
                expected: CAUSAL_RECEIPT_SCHEMA_VERSION,
                found: receipt.schema_version,
            });
        }
        Ok(receipt)
    }

    /// Serialize to TOML.
    pub fn to_toml(&self) -> Result<String, CausalError> {
        toml::to_string_pretty(self).map_err(|e| CausalError::Toml(e.to_string()))
    }

    /// Deserialize from TOML, validating schema version fail-closed.
    pub fn from_toml(toml_str: &str) -> Result<Self, CausalError> {
        let receipt: Self = toml::from_str(toml_str).map_err(|e| CausalError::Toml(e.to_string()))?;
        if receipt.schema_version != CAUSAL_RECEIPT_SCHEMA_VERSION {
            return Err(CausalError::StaleSchemaVersion {
                expected: CAUSAL_RECEIPT_SCHEMA_VERSION,
                found: receipt.schema_version,
            });
        }
        Ok(receipt)
    }

    /// Serialize to canonical bytes (JSON bytes representation).
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from canonical bytes with fail-closed schema check.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CausalError> {
        let receipt: Self = serde_json::from_slice(bytes).map_err(|e| CausalError::Json(e.to_string()))?;
        if receipt.schema_version != CAUSAL_RECEIPT_SCHEMA_VERSION {
            return Err(CausalError::StaleSchemaVersion {
                expected: CAUSAL_RECEIPT_SCHEMA_VERSION,
                found: receipt.schema_version,
            });
        }
        Ok(receipt)
    }
}
