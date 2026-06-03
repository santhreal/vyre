//! Subgroup collective semantics for the HashMap interpreter.
//!
//! These helpers operate on immutable invocation snapshots so collectives
//! observe a stable workgroup lane view.

#[cfg(feature = "subgroup-ops")]
use super::{
    memory::HashmapMemory,
    state::{HashmapInvocation, HashmapInvocationSnapshot},
    step::eval_expr_snapshot,
};
#[cfg(feature = "subgroup-ops")]
use crate::{subgroup::SubgroupSimulator, value::Value};
#[cfg(feature = "subgroup-ops")]
use smallvec::SmallVec;
#[cfg(feature = "subgroup-ops")]
use std::sync::OnceLock;
#[cfg(feature = "subgroup-ops")]
use vyre::ir::Expr;
#[cfg(feature = "subgroup-ops")]
use vyre::Error;

#[cfg(feature = "subgroup-ops")]
pub(crate) fn subgroup_simulator() -> &'static SubgroupSimulator {
    static SIMULATOR: OnceLock<SubgroupSimulator> = OnceLock::new();
    SIMULATOR.get_or_init(SubgroupSimulator::default)
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn subgroup_slice(
    snapshots: &[HashmapInvocationSnapshot],
    linear_local_index: u32,
) -> &[HashmapInvocationSnapshot] {
    let simulator = subgroup_simulator();
    let lane_index = linear_local_index as usize;
    let (start, end) = simulator.subgroup_bounds(snapshots.len(), lane_index);
    &snapshots[start..end]
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_subgroup_ballot(
    cond: &Expr,
    invocation: &HashmapInvocation<'_>,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<Value, Error> {
    let mask = collect_lane_bools(cond, invocation.linear_local_index, snapshots, memory)?;
    Ok(Value::U32(subgroup_simulator().ballot_slice(&mask)))
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_subgroup_shuffle(
    value: &Expr,
    lane: &Expr,
    invocation: &HashmapInvocation<'_>,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<Value, Error> {
    let values = collect_lane_u32s(
        value,
        invocation.linear_local_index,
        snapshots,
        memory,
        "subgroup_shuffle value is not a u32. Fix: use subgroup collectives with integer lanes only.",
    )?;
    let src_lanes = collect_lane_u32s(
        lane,
        invocation.linear_local_index,
        snapshots,
        memory,
        "subgroup_shuffle lane index is not a u32. Fix: use a scalar u32 lane argument.",
    )?;
    let shuffled = subgroup_simulator().shuffle(&values, &src_lanes);
    let local_offset = (invocation.linear_local_index as usize) % subgroup_simulator().width();
    Ok(Value::U32(shuffled.get(local_offset).copied().unwrap_or(0)))
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_subgroup_add(
    value: &Expr,
    invocation: &HashmapInvocation<'_>,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<Value, Error> {
    let values = collect_lane_values(value, invocation.linear_local_index, snapshots, memory)?;
    if values.iter().all(|value| matches!(value, Value::U32(_))) {
        let lanes = values
            .iter()
            .filter_map(Value::try_as_u32)
            .collect::<SmallVec<[u32; 32]>>();
        return Ok(Value::U32(subgroup_simulator().add(&lanes)));
    }
    if values.iter().all(|value| matches!(value, Value::Float(_))) {
        let sum = values.iter().fold(0.0f32, |acc, value| match value {
            Value::Float(lane) => crate::execution::typed_ops::canonical_f32(acc + (*lane as f32)),
            _ => acc,
        });
        return Ok(Value::Float(f64::from(sum)));
    }
    Err(Error::interp(
        "subgroup_add lanes have mixed or unsupported value types. Fix: cast every lane value to the same primitive u32 or f32 type before the subgroup collective.",
    ))
}

#[cfg(feature = "subgroup-ops")]
fn collect_lane_bools(
    expr: &Expr,
    linear_local_index: u32,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<SmallVec<[bool; 32]>, Error> {
    subgroup_slice(snapshots, linear_local_index)
        .iter()
        .map(|lane| eval_expr_snapshot(expr, lane, snapshots, memory).map(|value| value.truthy()))
        .collect()
}

#[cfg(feature = "subgroup-ops")]
fn collect_lane_u32s(
    expr: &Expr,
    linear_local_index: u32,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
    error: &'static str,
) -> Result<SmallVec<[u32; 32]>, Error> {
    subgroup_slice(snapshots, linear_local_index)
        .iter()
        .map(|lane| {
            eval_expr_snapshot(expr, lane, snapshots, memory)?
                .try_as_u32()
                .ok_or_else(|| Error::interp(error))
        })
        .collect()
}

#[cfg(feature = "subgroup-ops")]
fn collect_lane_values(
    expr: &Expr,
    linear_local_index: u32,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<SmallVec<[Value; 32]>, Error> {
    subgroup_slice(snapshots, linear_local_index)
        .iter()
        .map(|lane| eval_expr_snapshot(expr, lane, snapshots, memory))
        .collect()
}
