//! Registry-observing contracts, in one link unit.
//!
//! Every integration target in this crate links the whole operation registry,
//! its primitive catalog and every backend driver, so a second target costs a
//! second link of that surface. The modules below share this target instead of
//! paying for it once each.

#![forbid(unsafe_code)]

mod cli_docs;
mod handrolled_operations;
mod implementation_family_closure;
mod operation_schema;
mod operation_schema_placement;
mod registration_visibility;
