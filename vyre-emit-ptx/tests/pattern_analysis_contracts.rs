//! PTX pattern-analysis contracts, one module per analysis.

#[path = "pattern_analysis_contracts/audit.rs"]
mod audit;
#[path = "pattern_analysis_contracts/instruction_scheduling.rs"]
mod instruction_scheduling;
#[path = "pattern_analysis_contracts/ldmatrix_cp_async.rs"]
mod ldmatrix_cp_async;
#[path = "pattern_analysis_contracts/tensor_core_fragment.rs"]
mod tensor_core_fragment;
#[path = "pattern_analysis_contracts/vec_memory_fusion.rs"]
mod vec_memory_fusion;
