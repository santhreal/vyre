//! Contracts for lowering modules whose tests reach only the public API.
//!
//! WHY: these suites compiled into the library on every test build while
//! asserting nothing a consumer cannot call. One integration target keeps them
//! link-cheap and pins them to the surface the emitter crates see.

#[path = "lowering_contracts/alias_facts.rs"]
mod alias_facts;
#[path = "lowering_contracts/candidate_plan.rs"]
mod candidate_plan;
#[path = "lowering_contracts/descent_contract.rs"]
mod descent_contract;
#[path = "lowering_contracts/descriptor_builder.rs"]
mod descriptor_builder;
#[path = "lowering_contracts/operand_class.rs"]
mod operand_class;
#[path = "lowering_contracts/pattern_audit.rs"]
mod pattern_audit;
#[path = "lowering_contracts/program_stability_corpus.rs"]
mod program_stability_corpus;
#[path = "lowering_contracts/vec_pack.rs"]
mod vec_pack;
#[path = "lowering_contracts/verify_descriptor.rs"]
mod verify_descriptor;
