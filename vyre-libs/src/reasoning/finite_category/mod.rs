//! Finite categories: Yoneda embedding, adjoint pairs, Kan extensions.
//!
//! A category here is finite: a fixed object set plus a Hom-set cardinality
//! table. Morphisms are `u32` ids, identities are implicit, and composition is
//! not modelled, because every question this module answers is a cardinality
//! question. Functors carry an object map.
//!
//! Pass composition reasons over these. An adjunction `F ⊣ G` is the licence to
//! re-order an F-then-G pass pair into G-then-F; a Kan extension extends a
//! partially defined pass functor along a re-indexing functor; the Yoneda count
//! decides whether a pass tree maps into a representable family.
//!
//! Every entry point charges the analysis call counter once, so a caller that
//! reaches for a categorical fact is visible in telemetry whether or not the
//! fact was cheap to compute.

pub mod adjoint;
pub mod kan_extension;
pub mod yoneda;

pub use adjoint::{adjoint_pair, AdjointPair, FiniteFunctor};
pub use kan_extension::{kan_extension_at, kan_extension_table, KanDirection};
pub use yoneda::{natural_transformation_count, yoneda_embedding, FiniteCategory};
