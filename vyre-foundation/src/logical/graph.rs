//! Logical algorithm graph validation and canonical identity derivation.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::region::{
    LogicalAliasFacts, LogicalDependence, LogicalDependenceKind, LogicalEffects, LogicalExtent,
    LogicalIndexMap, LogicalLayout, LogicalProgramError, LogicalRegion, LogicalRegionKind,
    OrderingContract, OrderingSyncScope, ProgressContract, ScratchContract,
    LOGICAL_ALGORITHM_VERSION,
};
use crate::ir::{
    stats::{NODE_KIND_ALL_REDUCE, NODE_KIND_REDUCE_SCATTER, NODE_KIND_TILE_REDUCE},
    BufferAccess, GraphNodeId, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime,
};
use crate::logical_partition::{LogicalExchange, LogicalExchangeKind, LogicalPartitionFacts};
use crate::numeric::{
    graph_budget, region_contract, NumericContract, QuantizedContract, QuantizedRefusal,
    RegionArithmetic, RegionNumericFacts, ScalarFormat,
};
use crate::operation::OperationEffects;

/// A graph plus validated logical regions and schedule-free canonical identity.
#[derive(Debug)]
pub struct LogicalProgramGraph<'a> {
    graph: &'a ProgramGraph,
    regions: Vec<LogicalRegion>,
    exchanges: Vec<LogicalExchange>,
    semantic_wire: Vec<u8>,
}

impl<'a> LogicalProgramGraph<'a> {
    /// Validate and derive the logical algorithm stage.
    pub fn validate(
        graph: &'a ProgramGraph,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<Self, LogicalProgramError> {
        graph
            .analyze()
            .map_err(|error| LogicalProgramError::Graph(error.to_string()))?;

        let required = graph
            .values()
            .iter()
            .flat_map(|value| &value.contract.shape)
            .filter_map(|dim| match dim {
                ShapeDim::Symbol(symbol) => Some(symbol.as_str()),
                ShapeDim::Known(_) | ShapeDim::Unresolved | ShapeDim::Expr(_) => None,
            })
            .collect::<BTreeSet<_>>();
        for symbol in &required {
            if !bindings.contains_key(*symbol) {
                return Err(LogicalProgramError::MissingSymbol((*symbol).to_owned()));
            }
        }
        for symbol in bindings.keys() {
            if !required.contains(symbol.as_str()) {
                return Err(LogicalProgramError::UnexpectedSymbol(symbol.clone()));
            }
        }

        let mut regions = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            let (shape, source_value) =
                if let Some((port, value)) = node.output_ports.first().zip(node.outputs.first()) {
                    (port.contract.shape.as_slice(), Some(*value))
                } else if let Some(port) = node.inputs.first() {
                    (port.contract.shape.as_slice(), Some(port.value))
                } else {
                    (&[][..], None)
                };
            let extents = shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| {
                    let value = source_value.ok_or(LogicalProgramError::MissingDomain(node.id))?;
                    let axis = u32::try_from(axis)
                        .map_err(|_| LogicalProgramError::DomainRankOverflow(node.id))?;
                    match dim {
                        ShapeDim::Known(0) => Err(LogicalProgramError::UnresolvedExtent {
                            node: node.id,
                            value,
                            axis,
                        }),
                        ShapeDim::Known(bound) => Ok(LogicalExtent::Static(*bound)),
                        ShapeDim::Unresolved | ShapeDim::Expr(_) => {
                            Err(LogicalProgramError::UnresolvedExtent {
                                node: node.id,
                                value,
                                axis,
                            })
                        }
                        ShapeDim::Symbol(symbol) => {
                            let bound = bindings.get(symbol).copied().ok_or_else(|| {
                                LogicalProgramError::MissingSymbol(symbol.clone())
                            })?;
                            if bound == 0 {
                                return Err(LogicalProgramError::UnresolvedExtent {
                                    node: node.id,
                                    value,
                                    axis,
                                });
                            }
                            Ok(LogicalExtent::GraphValue {
                                value: value.0,
                                axis,
                                symbol: symbol.clone(),
                                bound,
                            })
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let max_points = extents.iter().try_fold(1u64, |product, extent| {
                product
                    .checked_mul(extent.bound())
                    .ok_or(LogicalProgramError::ExtentOverflow(node.id))
            })?;
            let row_major_strides = row_major_strides(&extents, node.id)?;
            let reads = node
                .inputs
                .iter()
                .filter(|input| input.contract.access != BufferAccess::WriteOnly)
                .map(|input| input.value.0)
                .collect::<Vec<_>>();
            let in_place_values = node
                .inputs
                .iter()
                .filter(|input| {
                    matches!(
                        input.contract.access,
                        BufferAccess::ReadWrite | BufferAccess::WriteOnly
                    )
                })
                .map(|input| input.value.0)
                .collect::<Vec<_>>();
            let mut writes = in_place_values.clone();
            writes.extend(
                node.output_ports
                    .iter()
                    .zip(&node.outputs)
                    .filter(|(output, _)| output.contract.access != BufferAccess::ReadOnly)
                    .map(|(_, value)| value.0),
            );
            writes.sort_unstable();
            writes.dedup();
            let retained_successors = node
                .output_ports
                .iter()
                .zip(&node.outputs)
                .filter_map(|(port, output)| {
                    port.retained_successor_of.map(|prior| (output.0, prior.0))
                })
                .collect::<Vec<_>>();
            let retained_state = !retained_successors.is_empty()
                || node.output_ports.iter().any(|port| {
                    matches!(
                        port.contract.lifetime,
                        ValueLifetime::Retained | ValueLifetime::Output
                    ) && port.retained_successor_of.is_some()
                })
                || node.inputs.iter().any(|port| {
                    port.contract.lifetime == ValueLifetime::Retained
                        && port.contract.access == BufferAccess::ReadWrite
                });
            let program_effects = OperationEffects::from_program(&node.program);
            let reduction_mask =
                NODE_KIND_ALL_REDUCE | NODE_KIND_REDUCE_SCATTER | NODE_KIND_TILE_REDUCE;
            let kind = if retained_state {
                LogicalRegionKind::RetainedState
            } else if node.program.stats().has_any_node_kind(reduction_mask) {
                LogicalRegionKind::Reduction
            } else if node.program.stats().control_flow_count > 0
                || program_effects.atomics
                || program_effects.synchronizes
            {
                LogicalRegionKind::Sequential
            } else {
                LogicalRegionKind::Parallel
            };

            let mut dependence_values = BTreeMap::<GraphNodeId, Vec<u32>>::new();
            for input in &node.inputs {
                let value = graph
                    .values()
                    .get(input.value.0 as usize)
                    .ok_or(LogicalProgramError::MissingDomainValue(input.value))?;
                if let Some(predecessor) = value.producer {
                    if predecessor >= node.id {
                        return Err(LogicalProgramError::CyclicDomain {
                            node: node.id,
                            predecessor,
                        });
                    }
                    dependence_values
                        .entry(predecessor)
                        .or_default()
                        .push(input.value.0);
                }
            }
            let mut dependencies = Vec::with_capacity(dependence_values.len());
            for (predecessor, values) in dependence_values {
                let bytes = crate::logical_partition::value_bytes(graph, bindings, &values)
                    .map_err(LogicalProgramError::Exchange)?;
                dependencies.push(LogicalDependence {
                    predecessor,
                    kind: if retained_state {
                        LogicalDependenceKind::RetainedState
                    } else {
                        LogicalDependenceKind::Flow
                    },
                    values,
                    bytes,
                });
            }
            let inputs_disjoint = node
                .inputs
                .iter()
                .map(|input| input.value.0)
                .collect::<BTreeSet<_>>()
                .len()
                == node.inputs.len();
            let outputs_disjoint = node
                .outputs
                .iter()
                .map(|value| value.0)
                .collect::<BTreeSet<_>>()
                .len()
                == node.outputs.len();
            if !inputs_disjoint || !outputs_disjoint {
                return Err(LogicalProgramError::IncompatibleAliases(node.id));
            }
            let storage_order = (0..shape.len())
                .map(|axis| {
                    u32::try_from(axis)
                        .map_err(|_| LogicalProgramError::DomainRankOverflow(node.id))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let reduction_axes = if kind == LogicalRegionKind::Reduction {
                storage_order.clone()
            } else {
                Vec::new()
            };
            let in_place_reads = node
                .inputs
                .iter()
                .any(|input| input.contract.access == BufferAccess::ReadWrite);
            let partition = crate::logical_partition::partition_facts(
                &extents.iter().map(LogicalExtent::bound).collect::<Vec<_>>(),
                &reduction_axes,
                matches!(
                    kind,
                    LogicalRegionKind::Sequential | LogicalRegionKind::RetainedState
                ),
                retained_state,
                program_effects.atomics,
                in_place_reads,
            );
            let written_bytes = crate::logical_partition::value_bytes(graph, bindings, &writes)
                .map_err(LogicalProgramError::Exchange)?;
            let quantized = quantized_input(node.inputs.iter().map(|input| &input.contract.dtype))
                .map_err(|refusal| LogicalProgramError::Numeric {
                    node: node.id,
                    reason: refusal.to_string(),
                })?;
            let region = region_contract(&RegionNumericFacts {
                input: promoted_format(node.inputs.iter().map(|input| &input.contract.dtype)),
                output: promoted_format(node.output_ports.iter().map(|port| &port.contract.dtype)),
                arithmetic: match kind {
                    LogicalRegionKind::Parallel
                    | LogicalRegionKind::SegmentedMap
                    | LogicalRegionKind::Window
                    | LogicalRegionKind::RaggedExtent => RegionArithmetic::Pointwise,
                    LogicalRegionKind::Reduction
                    | LogicalRegionKind::Scan
                    | LogicalRegionKind::PartialResultJoin => RegionArithmetic::Reduction {
                        terms: reduced_points(&extents, &reduction_axes),
                    },
                    LogicalRegionKind::Sequential
                    | LogicalRegionKind::RetainedState
                    | LogicalRegionKind::RecurrentState => {
                        RegionArithmetic::Recurrence { steps: max_points }
                    }
                },
                atomics: program_effects.atomics,
                reorderable: crate::algebraic_reordering::reordering_class(&node.program)
                    .permits_reordering(),
            })
            .map_err(|refusal| LogicalProgramError::Numeric {
                node: node.id,
                reason: refusal.to_string(),
            })?;
            let requantized =
                quantized_input(node.output_ports.iter().map(|port| &port.contract.dtype))
                    .map_err(|refusal| LogicalProgramError::Numeric {
                        node: node.id,
                        reason: refusal.to_string(),
                    })?;
            let numeric = numeric_budget(&quantized, &requantized, region).map_err(|refusal| {
                LogicalProgramError::Numeric {
                    node: node.id,
                    reason: refusal.to_string(),
                }
            })?;
            let workgroup_scratch_bytes = node
                .program
                .buffers()
                .iter()
                .filter(|b| b.access() == BufferAccess::Workgroup)
                .map(|b| u64::from(b.count()) * b.element().size_bytes().unwrap_or(4) as u64)
                .sum::<u64>();
            let scratch = ScratchContract {
                workgroup_scratch_bytes,
                partition_scratch_bytes: 0,
                reusable: true,
            };
            let progress = ProgressContract {
                max_iterations: max_points.max(1),
                monotone_progress: true,
                guaranteed_termination: true,
            };
            let causal_ordering = matches!(
                kind,
                LogicalRegionKind::Sequential
                    | LogicalRegionKind::RetainedState
                    | LogicalRegionKind::RecurrentState
                    | LogicalRegionKind::Scan
            );
            let sync_scope = if program_effects.synchronizes
                || matches!(kind, LogicalRegionKind::Reduction | LogicalRegionKind::Scan)
            {
                OrderingSyncScope::Workgroup
            } else if retained_state {
                OrderingSyncScope::RetainedEpoch
            } else {
                OrderingSyncScope::None
            };
            let ordering = OrderingContract {
                causal_ordering,
                sync_scope,
            };
            regions.push(LogicalRegion {
                node: node.id,
                name: node.name.clone(),
                kind,
                extents,
                index_map: LogicalIndexMap {
                    axes: (0..shape.len()).map(|axis| format!("axis{axis}")).collect(),
                    row_major_strides: row_major_strides.clone(),
                },
                layout: LogicalLayout {
                    storage_order,
                    strides: row_major_strides,
                    contiguous: true,
                },
                reduction_axes,
                aliases: LogicalAliasFacts {
                    retained_successors,
                    in_place_values,
                    inputs_disjoint,
                    outputs_disjoint,
                },
                dependencies,
                effects: LogicalEffects {
                    reads,
                    writes,
                    retained_state,
                    atomics: program_effects.atomics,
                    synchronizes: program_effects.synchronizes,
                },
                partition,
                written_bytes,
                max_points,
                numeric,
                segment: None,
                window: None,
                recurrence: None,
                partial_join: None,
                scratch,
                progress,
                ordering,
            });
        }

        #[derive(Serialize)]
        struct IdentityDependence<'b> {
            predecessor: u32,
            values: &'b [u32],
            bytes: u64,
            kind: LogicalDependenceKind,
        }
        #[derive(Serialize)]
        struct IdentityRegion<'b> {
            node: u32,
            name: &'b str,
            kind: LogicalRegionKind,
            extents: &'b [LogicalExtent],
            index_map: &'b LogicalIndexMap,
            layout: &'b LogicalLayout,
            reduction_axes: &'b [u32],
            aliases: &'b LogicalAliasFacts,
            dependencies: Vec<IdentityDependence<'b>>,
            effects: &'b LogicalEffects,
            partition: &'b LogicalPartitionFacts,
            written_bytes: u64,
            max_points: u64,
            scratch: &'b ScratchContract,
            progress: &'b ProgressContract,
            ordering: &'b OrderingContract,
        }
        #[derive(Serialize)]
        struct IdentityExchange<'b> {
            node: u32,
            kind: LogicalExchangeKind,
            group: u32,
            combine: Option<crate::ir::CollectiveOp>,
            values: &'b [u32],
            bytes: u64,
        }
        #[derive(Serialize)]
        struct Identity<'b> {
            version: u16,
            regions: Vec<IdentityRegion<'b>>,
            exchanges: Vec<IdentityExchange<'b>>,
            graph: &'b [u8],
        }
        let graph_wire = graph
            .logical_wire()
            .map_err(|error| LogicalProgramError::CanonicalGraph(error.to_string()))?;
        let identity_regions = regions
            .iter()
            .map(|region| IdentityRegion {
                node: region.node.0,
                name: &region.name,
                kind: region.kind,
                extents: &region.extents,
                index_map: &region.index_map,
                layout: &region.layout,
                reduction_axes: &region.reduction_axes,
                aliases: &region.aliases,
                dependencies: region
                    .dependencies
                    .iter()
                    .map(|dependence| IdentityDependence {
                        predecessor: dependence.predecessor.0,
                        values: &dependence.values,
                        bytes: dependence.bytes,
                        kind: dependence.kind,
                    })
                    .collect(),
                effects: &region.effects,
                partition: &region.partition,
                written_bytes: region.written_bytes,
                max_points: region.max_points,
                scratch: &region.scratch,
                progress: &region.progress,
                ordering: &region.ordering,
            })
            .collect();
        let exchanges = crate::logical_partition::exchanges(graph, bindings)
            .map_err(LogicalProgramError::Exchange)?;
        let semantic_wire = serde_json::to_vec(&Identity {
            version: LOGICAL_ALGORITHM_VERSION,
            regions: identity_regions,
            exchanges: exchanges
                .iter()
                .map(|exchange| IdentityExchange {
                    node: exchange.node.0,
                    kind: exchange.kind,
                    group: exchange.group,
                    combine: exchange.combine,
                    values: &exchange.values,
                    bytes: exchange.bytes,
                })
                .collect(),
            graph: &graph_wire,
        })
        .map_err(|error| LogicalProgramError::Identity(error.to_string()))?;

        Ok(Self {
            graph,
            regions,
            exchanges,
            semantic_wire,
        })
    }

    /// Borrow the whole-program graph this logical stage was derived from.
    #[must_use]
    pub const fn graph(&self) -> &'a ProgramGraph {
        self.graph
    }

    /// Borrow validated logical regions in graph-node order.
    #[must_use]
    pub fn regions(&self) -> &[LogicalRegion] {
        &self.regions
    }

    /// Semantic exchanges the graph states, in graph-node order.
    #[must_use]
    pub fn exchanges(&self) -> &[LogicalExchange] {
        &self.exchanges
    }

    /// Canonical schedule-free identity bytes.
    #[must_use]
    pub fn semantic_wire(&self) -> &[u8] {
        &self.semantic_wire
    }

    /// The numeric contract one graph value carries.
    ///
    /// The regions that feed the value are composed in dependence order, so the
    /// budget is what the value accumulated along its own chain rather than the
    /// widest region in the graph. A value no region produces carries the exact
    /// contract: it is what the caller supplied.
    ///
    /// # Errors
    ///
    /// Returns [`LogicalProgramError::Numeric`] when two regions in the chain
    /// cannot be composed, which is a region reading a format the region before
    /// it does not produce.
    pub fn value_budget(
        &self,
        value: GraphValueId,
    ) -> Result<NumericContract, LogicalProgramError> {
        let Some(producer) = self
            .graph
            .values()
            .get(value.0 as usize)
            .and_then(|value| value.producer)
        else {
            return Ok(NumericContract::EXACT);
        };
        let mut chain = BTreeSet::new();
        self.collect_ancestors(producer, &mut chain);
        let contracts = chain
            .iter()
            .filter_map(|node| self.region(*node))
            .map(|region| &region.numeric);
        crate::numeric::graph_budget(contracts).map_err(|refusal| LogicalProgramError::Numeric {
            node: producer,
            reason: refusal.to_string(),
        })
    }

    /// The numeric contract every caller-visible output carries.
    ///
    /// # Errors
    ///
    /// Returns the first [`LogicalProgramError::Numeric`] one output produced.
    pub fn output_budgets(
        &self,
    ) -> Result<Vec<(GraphValueId, NumericContract)>, LogicalProgramError> {
        self.graph
            .values()
            .iter()
            .filter(|value| value.contract.lifetime == ValueLifetime::Output)
            .map(|value| Ok((value.id, self.value_budget(value.id)?)))
            .collect()
    }

    /// The region one graph node states.
    #[must_use]
    pub fn region(&self, node: GraphNodeId) -> Option<&LogicalRegion> {
        self.regions.iter().find(|region| region.node == node)
    }

    /// Collect `node` and every region it transitively depends on.
    fn collect_ancestors(&self, node: GraphNodeId, chain: &mut BTreeSet<GraphNodeId>) {
        if !chain.insert(node) {
            return;
        }
        let Some(region) = self.region(node) else {
            return;
        };
        for dependence in &region.dependencies {
            self.collect_ancestors(dependence.predecessor, chain);
        }
    }

    /// Compute structural sharing metrics over the underlying ProgramGraph.
    #[must_use]
    pub fn structural_sharing_metrics(&self) -> crate::ir::ProgramGraphSharingMetrics {
        self.graph.structural_sharing_metrics()
    }

    /// Number of distinct logical region bodies in the graph.
    #[must_use]
    pub fn canonical_region_count(&self) -> usize {
        let mut unique_regions = rustc_hash::FxHashSet::default();
        for region in &self.regions {
            unique_regions.insert((
                region.kind,
                region.extents.clone(),
                region.reduction_axes.clone(),
                region.effects.retained_state,
                region.effects.atomics,
                region.effects.synchronizes,
            ));
        }
        unique_regions.len()
    }
}

/// The format a region computes in, given the formats it reads or writes.
///
/// A rounding format wins over an exact one because that is the promotion an
/// operation performs, and among rounding formats the finest wins because a
/// value held in it is not made coarser by being read. A quantized value is
/// stored packed and computed dequantized, so it answers the logical format its
/// contract states: reading it through the storage format would price an INT4
/// region as exact arithmetic.
fn promoted_format<'b>(
    types: impl Iterator<Item = &'b crate::ir::DataType>,
) -> Option<ScalarFormat> {
    types
        .filter_map(arithmetic_format)
        .fold(None, |best, next| {
            let Some(best) = best else {
                return Some(next);
            };
            Some(match (best.ulp_fraction(), next.ulp_fraction()) {
                (None, Some(_)) => next,
                (Some(_), None) => best,
                (Some(left), Some(right)) if right < left => next,
                _ => best,
            })
        })
}

/// The scalar format one value is computed in.
fn arithmetic_format(dtype: &crate::ir::DataType) -> Option<ScalarFormat> {
    QuantizedContract::of(dtype).map_or_else(
        || ScalarFormat::of(dtype),
        |contract| Some(contract.logical),
    )
}

/// The quantized contract every quantized input of one region states.
///
/// A region reading two quantized values packed differently cannot read both
/// with one lane law, so it is refused here rather than reinterpreting the
/// bytes of one of them.
fn quantized_input<'b>(
    types: impl Iterator<Item = &'b crate::ir::DataType>,
) -> Result<Option<QuantizedContract>, QuantizedRefusal> {
    let mut stated: Option<QuantizedContract> = None;
    for contract in types.filter_map(QuantizedContract::of) {
        contract.check()?;
        match &stated {
            None => stated = Some(contract),
            Some(first) if first.propagates_to(&contract) => {}
            Some(first) => {
                return Err(QuantizedRefusal::LayoutsDisagree {
                    first: first.to_string(),
                    second: contract.to_string(),
                })
            }
        }
    }
    Ok(stated)
}

/// What one region computes, given what it reads and what it writes.
///
/// Reading a quantized value costs the step of the grid it was placed on, and
/// writing one places it on a grid: onto a new one when the region read nothing
/// quantized, and onto another one when the region read a different grid. A
/// conversion that only moves fields inside their container or reads a different
/// sidecar changes no value, so it is priced at what it is: nothing. The composed
/// budget is what a stated ceiling is checked against, which is how a caller
/// learns that a quantizing chain is wider than the compute inside it.
fn numeric_budget(
    read: &Option<QuantizedContract>,
    written: &Option<QuantizedContract>,
    region: NumericContract,
) -> Result<NumericContract, crate::numeric::ContractRefusal> {
    let mut composed = match read {
        None => region,
        Some(contract) => graph_budget([&contract.numeric(), &region])?,
    };
    if let Some(target) = written {
        let measure = read.as_ref().map_or_else(
            || target.dequantization_measure(),
            |source| source.conversion(target).measure(target),
        );
        composed = graph_budget([&composed, &region.with_measure(measure)])?;
    }
    Ok(composed)
}

/// The number of points one reduction combines into a single output point.
fn reduced_points(extents: &[LogicalExtent], axes: &[u32]) -> u64 {
    axes.iter()
        .filter_map(|axis| extents.get(*axis as usize))
        .fold(1u64, |product, extent| {
            product.saturating_mul(extent.bound())
        })
}

fn row_major_strides(
    extents: &[LogicalExtent],
    node: GraphNodeId,
) -> Result<Vec<u64>, LogicalProgramError> {
    let mut strides = vec![0; extents.len()];
    let mut stride = 1u64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = stride;
        let value = extent.bound();
        stride = stride
            .checked_mul(value)
            .ok_or(LogicalProgramError::ExtentOverflow(node))?;
    }
    Ok(strides)
}
