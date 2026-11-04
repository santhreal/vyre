//! Contracts for lowering modules whose tests reach only the public API.
//!
//! WHY: these suites compiled into the library on every test build while
//! asserting nothing a consumer cannot call. One integration target keeps them
//! link-cheap and pins them to the surface the emitter crates see.

mod alias_facts;
mod candidate_plan;
mod descent_contract;
mod descriptor_builder;
mod literal_serde;
mod operand_class;
mod pattern_audit;
mod program_stability_corpus;
mod vec_pack;
mod verify_descriptor;
