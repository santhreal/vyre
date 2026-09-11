//! Numeric contracts: what a result is allowed to be, and what a schedule may
//! do to it.
//!
//! A schedule decides where a value is computed, in what order combines land,
//! what a reduction accumulates in, and whether an approximate instruction is
//! selected. Each of those changes the result. A contract states the change a
//! caller admits, composition states the change a whole graph accumulates, and
//! legality refuses the schedule that cannot prove it stays inside the stated
//! bound. Nothing here measures a device: the proof is carried out on the IR,
//! before lowering, so a refusal is a compile-time answer.

mod contract;
mod format;
mod quantized;
mod range;
mod region;

pub use contract::{
    Approximation, AtomicOrderSensitivity, ContractRefusal, Determinism, ErrorMeasure,
    NumericContract, Reassociation, NUMERIC_CONTRACT_VERSION,
};
pub use format::ScalarFormat;
pub use quantized::{
    CalibrationIdentity, FieldTarget, GroupAxis, PackedField, PackingOrder, QuantizedContract,
    QuantizedConversion, QuantizedRefusal, CALIBRATION_IDENTITY_VERSION,
    QUANTIZED_CONTRACT_VERSION,
};
pub use range::{prove, MagnitudeRange, NumericChoice, RangeProof};
pub use region::{
    budget_admits, graph_budget, region_contract, reordering_admitted, RegionArithmetic,
    RegionNumericFacts,
};
