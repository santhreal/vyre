//! The live inventory registries, with every crate that submits into them linked.
//!
//! WHY: `inventory` registrations live in the object file of the crate that
//! declares them, and a linker keeps that object only when a symbol inside it is
//! referenced. `use vyre_libs as _;` names a crate without referencing any symbol
//! in it, and `std::hint::black_box(SOME_BACKEND_ID)` reads a `const` that inlines
//! at the use site, so neither is a link anchor. A binary whose only tie to a
//! submitting crate was one of those shapes read a registry shorter than the tree
//! declares, and every count, document and rule agreed with itself while judging
//! a partial tree: three registry rules once iterated zero operation
//! registrations and passed, and backend evidence reported a shorter linked set
//! than the drivers it named.
//!
//! Reading a registry through this crate makes the reference real. Each accessor
//! calls a real symbol in every source crate, so the registrations are linked
//! into whatever binary reads them, and the floor per source is asserted rather
//! than assumed:
//!
//! - [`operation::live_operation_registry`] over the operation registry, whose
//!   sources are the operation-owning crates.
//! - [`backend::live_backend_registry`] over the backend registry, whose sources
//!   are the concrete driver crates.
//!
//! This crate is the one place that names those crates for linkage. A consumer
//! enables the cargo features naming the registries it reads and the drivers it
//! legitimately depends on, and each accessor reports exactly the set it linked,
//! so a narrower consumer states its set instead of silently accepting a shorter
//! registry.

#![forbid(unsafe_code)]

pub mod backend;
#[cfg(feature = "operations")]
pub mod operation;
