//! What every cost field means, in what unit, and where its weight came from.
//!
//! A cost model whose fields are bare integers cannot be audited: a reader
//! cannot tell nanoseconds from bytes, an estimate from a count, or a weight a
//! recording fixed from one somebody preferred. Each field of
//! [`CostBreakdown`](super::CostBreakdown) therefore has one row here, and
//! `every_cost_field_states_its_unit_and_provenance` derives the field set from
//! the serialized shape at run time, so a new field with no row turns the suite
//! red instead of entering the total unexplained.

use super::CostBreakdown;

/// Unit a cost field is measured in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostUnit {
    /// A count of program or plan objects.
    Count,
    /// Bytes of memory traffic or capacity.
    Bytes,
    /// 32-bit registers per invocation.
    Registers,
    /// Nanoseconds of expected device time.
    Nanoseconds,
}

/// Whether a field enters the ranked total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostTermRole {
    /// Recorded for audit and excluded from the total.
    Evidence,
    /// Summed into the total that candidate selection minimizes.
    Charged,
}

/// One field of [`CostBreakdown`](super::CostBreakdown): its unit, its role in
/// the total, and the fact or recording that fixes its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostTerm {
    /// Serialized field name.
    pub field: &'static str,
    /// Unit the field is measured in.
    pub unit: CostUnit,
    /// Whether the field is charged to the total.
    pub role: CostTermRole,
    /// Where the number comes from: a device fact, a recorded figure, or the
    /// graph itself.
    pub provenance: &'static str,
}

const fn term(
    field: &'static str,
    unit: CostUnit,
    role: CostTermRole,
    provenance: &'static str,
) -> CostTerm {
    CostTerm {
        field,
        unit,
        role,
        provenance,
    }
}

/// Every field of [`CostBreakdown`](super::CostBreakdown), in declaration order.
pub const TERMS: &[CostTerm] = &[
    term(
        "semantic_work",
        CostUnit::Count,
        CostTermRole::Evidence,
        "ProgramStats::node_count summed over the graph; equal for every candidate",
    ),
    term(
        "launches",
        CostUnit::Count,
        CostTermRole::Evidence,
        "generated kernel launches the candidate topology needs",
    ),
    term(
        "materializations",
        CostUnit::Count,
        CostTermRole::Evidence,
        "data edges crossing a fusion-group boundary",
    ),
    term(
        "materialized_bytes",
        CostUnit::Bytes,
        CostTermRole::Evidence,
        "packed byte length of the values on those edges",
    ),
    term(
        "live_value_peak",
        CostUnit::Registers,
        CostTermRole::Evidence,
        "ProgramStats::register_pressure_estimate summed over the worst group",
    ),
    term(
        "shared_scratch_bytes",
        CostUnit::Bytes,
        CostTermRole::Evidence,
        "workgroup buffer declarations of the worst group, unioned by name",
    ),
    term(
        "occupancy_passes_peak",
        CostUnit::Count,
        CostTermRole::Evidence,
        "DeviceFacts occupancy budgets divided into the worst group's demand",
    ),
    term(
        "planned_peak_bytes",
        CostUnit::Bytes,
        CostTermRole::Evidence,
        "the allocation plan's liveness peak over this grouping's stages",
    ),
    term(
        "instructions",
        CostUnit::Count,
        CostTermRole::Evidence,
        "ProgramStats::instruction_count summed over the graph",
    ),
    term(
        "tensor_ops",
        CostUnit::Count,
        CostTermRole::Evidence,
        "ProgramStats::tensor_op_count summed over the graph",
    ),
    term(
        "barriers",
        CostUnit::Count,
        CostTermRole::Evidence,
        "ProgramStats::barrier_count summed over the graph",
    ),
    term(
        "grid_syncs",
        CostUnit::Count,
        CostTermRole::Evidence,
        "stated grid rendezvous plus one per resident-partition stage boundary",
    ),
    term(
        "divergent_regions",
        CostUnit::Count,
        CostTermRole::Evidence,
        "CostCertificate::divergence_score summed over the graph",
    ),
    term(
        "spill_registers_peak",
        CostUnit::Registers,
        CostTermRole::Evidence,
        "worst group's live values above DeviceFacts::registers_per_invocation",
    ),
    term(
        "cache_resident_bytes",
        CostUnit::Bytes,
        CostTermRole::Evidence,
        "replayed bytes within DeviceFacts::cache_capacity_bytes",
    ),
    term(
        "reported_spill_bytes",
        CostUnit::Bytes,
        CostTermRole::Evidence,
        "target-reported local spill per invocation times launched invocations",
    ),
    term(
        "launch_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "DeviceFacts::per_launch_overhead_ns, or the cheapest recorded dispatch",
    ),
    term(
        "materialization_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "materialized bytes at the calibrated or peak bandwidth fact",
    ),
    term(
        "occupancy_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "replayed bytes the cache does not serve, at the same bandwidth fact",
    ),
    term(
        "instruction_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "DeviceFacts::compute_throughput_ops_per_ns; omitted when unmeasured",
    ),
    term(
        "tensor_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "DeviceFacts::tensor_throughput_ops_per_ns; omitted when unmeasured",
    ),
    term(
        "synchronization_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "DeviceFacts::barrier_ns and grid_sync_ns; omitted when unmeasured",
    ),
    term(
        "divergence_ns",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "idle lanes of a gated region at the compute rate; omitted when unmeasured",
    ),
    term(
        "total",
        CostUnit::Nanoseconds,
        CostTermRole::Charged,
        "sum of every charged term",
    ),
];

impl CostBreakdown {
    /// The unit and provenance row for one serialized field name.
    #[must_use]
    pub fn term(field: &str) -> Option<&'static CostTerm> {
        TERMS.iter().find(|term| term.field == field)
    }
}

#[cfg(test)]
mod tests {
    use super::{CostTermRole, CostUnit, TERMS};
    use crate::cost::CostBreakdown;

    /// Serialized field names of the cost breakdown, read from the shape itself
    /// rather than from a list somebody keeps up to date.
    fn serialized_fields() -> Vec<String> {
        let value = serde_json::to_value(CostBreakdown::default())
            .expect("the cost breakdown must serialize");
        value
            .as_object()
            .expect("the cost breakdown must serialize as an object")
            .keys()
            .cloned()
            .collect()
    }

    /// WHY: a cost field with no stated unit cannot be audited, and one with no
    /// stated provenance is a preference wearing a number. The field set comes
    /// from the serialized shape at run time, so adding a field without a row
    /// fails here instead of entering the total unexplained.
    #[test]
    fn every_cost_field_states_its_unit_and_provenance() {
        for field in serialized_fields() {
            let term = CostBreakdown::term(&field)
                .unwrap_or_else(|| panic!("cost field `{field}` has no unit and provenance row"));
            assert!(
                !term.provenance.is_empty(),
                "cost field `{field}` states an empty provenance"
            );
        }
    }

    /// WHY: a row for a field that no longer exists reads as documentation of
    /// the current model and documents a removed one.
    #[test]
    fn every_provenance_row_names_a_field_that_exists() {
        let fields = serialized_fields();
        for term in TERMS {
            assert!(
                fields.iter().any(|field| field == term.field),
                "provenance row `{}` names no field of the cost breakdown",
                term.field
            );
        }
    }

    /// WHY: the total is the number selection minimizes, so a charged term that
    /// is not nanoseconds is a unit error that silently reweights every ranking.
    #[test]
    fn every_charged_term_is_nanoseconds() {
        for term in TERMS {
            if term.role == CostTermRole::Charged {
                assert_eq!(
                    term.unit,
                    CostUnit::Nanoseconds,
                    "charged term `{}` is not measured in nanoseconds",
                    term.field
                );
            }
        }
    }
}
