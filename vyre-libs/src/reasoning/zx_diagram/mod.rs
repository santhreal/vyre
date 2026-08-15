//! ZX-diagram rewriting.
//!
//! ZX is a graphical language for linear maps. A diagram is an undirected
//! multigraph whose vertices, called spiders, each carry a color (Z or X) and a
//! phase. The rewrite rules the optimizer applies for pattern simplification
//! live here:
//!
//! * spider fusion (S1): adjacent same-color spiders merge, summing phases.
//! * identity removal (S2): a phase-zero spider whose two edges run to two
//!   other same-color spiders is dropped, splicing those edges into one.
//! * color change (H): conjugation by a Hadamard turns a Z-spider into an
//!   X-spider and back.
//!
//! [`simplified_diagram`] is the joint fixpoint of the first two.
//!
//! No floating point. A phase is a numerator over the diagram's `phase_denom`,
//! and two phases are equal when their numerators agree modulo it. A caller
//! picks `phase_denom = 8` for the Clifford+T fragment, `4` for Clifford only,
//! and `2` for the simplest stabilizer fragment.
//!
//! The optimizer's pattern-simplification pass treats commutative same-color
//! operators in the IR Region tree as Z spiders and folds them here.

pub mod rewrite;

pub use rewrite::{
    color_change, identity_removal, simplified_diagram, spider_fusion, ZxColor, ZxDiagram, ZxSpider,
};
