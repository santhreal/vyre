//! Contracts for reference modules whose tests reach only the public API.
//!
//! WHY: these suites lived beside the code they exercise but touched nothing
//! private, so they compiled into the library on every test build and could not
//! catch a symbol dropped from the public surface. One integration target keeps
//! them link-cheap and pins them to the API a consumer sees.

#[path = "core_contracts/dialect_dispatch.rs"]
mod dialect_dispatch;
#[path = "core_contracts/ieee754.rs"]
mod ieee754;
#[path = "core_contracts/subgroup.rs"]
mod subgroup;
#[path = "core_contracts/value.rs"]
mod value;
