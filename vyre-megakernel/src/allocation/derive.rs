//! Deriving the planner's per-value facts from the closed logical stage.
//!
//! Every fact here already exists: the byte totals and stage spans are the ones
//! the artifact's resource rows report, and the alias, effect and layout facts
//! are the ones logical validation closed. Nothing is re-derived, so a placement
//! and the resource row for the same value can never state different bytes or a
//! different live range.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::ShapeDim;
use vyre_foundation::logical::LogicalProgramGraph;

use super::pack::ValueFact;
use super::PlacementLayout;
use crate::error::{failure, overflow, CompileError, CompilerFailureKind};
use crate::identity::ArtifactNodeId;
use crate::schema::ResourceRecord;

/// Per-value planner facts for every value the graph declares.
///
/// # Errors
///
/// Returns [`CompileError`] when a resource row names a value the graph does not
/// declare, an element width is not statable, or a stride product overflows.
pub(crate) fn value_facts(
    logical: &LogicalProgramGraph<'_>,
    resources: &[ResourceRecord],
    bindings: &BTreeMap<String, u64>,
) -> Result<Vec<ValueFact>, CompileError> {
    let graph = logical.graph();
    let mut synchronized = BTreeSet::<u32>::new();
    let mut in_place = BTreeSet::<u32>::new();
    let mut region_of_node = BTreeMap::<u32, usize>::new();
    for (index, region) in logical.regions().iter().enumerate() {
        region_of_node.insert(region.node.0, index);
        in_place.extend(region.aliases.in_place_values.iter().copied());
        if region.effects.synchronizes || region.effects.atomics {
            synchronized.extend(region.effects.reads.iter().copied());
            synchronized.extend(region.effects.writes.iter().copied());
        }
    }

    resources
        .iter()
        .map(|resource| {
            let value = graph
                .values()
                .iter()
                .find(|candidate| candidate.id.0 == resource.value.0)
                .ok_or_else(|| {
                    failure(
                        CompilerFailureKind::InvalidProgram,
                        format!("artifact.resources[{}]", resource.value.0),
                        "resource row names a value the graph does not declare",
                        "record one resource row per declared graph value",
                    )
                })?;
            let element_bytes = value
                .contract
                .dtype
                .packed_size_bytes(1)
                .map_err(|message| {
                    overflow(format!("graph.values[{}].dtype", value.name), message)
                })?
                .and_then(|bytes| u32::try_from(bytes).ok())
                .ok_or_else(|| {
                    failure(
                        CompilerFailureKind::UnsizedResource,
                        format!("graph.values[{}].dtype", value.name),
                        "element representation has no fixed packed byte width",
                        "resolve the representation to a fixed-width typed value before compilation",
                    )
                })?;
            let strides = row_major_strides(value.name.as_str(), &value.contract.shape, bindings)?;
            let layout = value
                .producer
                .and_then(|producer| region_of_node.get(&producer.0))
                .map(|index| &logical.regions()[*index].layout)
                .filter(|declared| declared.strides.len() == strides.len())
                .map_or_else(
                    || PlacementLayout {
                        element_bytes,
                        storage_order: (0..axis_count(strides.len())).collect(),
                        strides: strides.clone(),
                        contiguous: true,
                    },
                    |declared| PlacementLayout {
                        element_bytes,
                        storage_order: declared.storage_order.clone(),
                        strides: declared.strides.clone(),
                        contiguous: declared.contiguous,
                    },
                );
            Ok(ValueFact {
                value: resource.value,
                producer: value.producer.map(|producer| ArtifactNodeId(producer.0)),
                bytes: resource.byte_count,
                element_bytes,
                lifetime: resource.lifetime,
                retained_predecessor: resource.retained_predecessor,
                first_stage: resource.first_stage,
                last_stage: resource.last_stage,
                produced: value.producer.is_some(),
                consumer_count: u32::try_from(value.consumers.len()).unwrap_or(u32::MAX),
                synchronized: synchronized.contains(&value.id.0),
                in_place: in_place.contains(&value.id.0),
                layout,
            })
        })
        .collect()
}

/// Element strides of one resolved shape in row-major order.
fn row_major_strides(
    name: &str,
    shape: &[ShapeDim],
    bindings: &BTreeMap<String, u64>,
) -> Result<Vec<u64>, CompileError> {
    let mut strides = vec![1u64; shape.len()];
    let mut running = 1u64;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = running;
        let extent = match dim {
            ShapeDim::Known(extent) => *extent,
            ShapeDim::Symbol(symbol) => *bindings.get(symbol).ok_or_else(|| {
                failure(
                    CompilerFailureKind::MissingSymbol,
                    format!("graph.values[{name}].shape"),
                    format!("symbolic extent `{symbol}` has no exact binding"),
                    "bind every symbolic graph dimension before compilation",
                )
            })?,
        };
        running = running.checked_mul(extent).ok_or_else(|| {
            overflow(
                format!("graph.values[{name}].shape"),
                "stride product exceeds u64",
            )
        })?;
    }
    Ok(strides)
}

fn axis_count(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}
