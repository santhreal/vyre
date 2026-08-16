//! SPIR-V pattern-analysis contracts, one module per analysis.

#[path = "pattern_analysis_contracts/audit.rs"]
mod audit;
#[path = "pattern_analysis_contracts/subgroup_capabilities.rs"]
mod subgroup_capabilities;
#[path = "pattern_analysis_contracts/workgroup_size_validation.rs"]
mod workgroup_size_validation;
