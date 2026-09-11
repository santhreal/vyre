//! Language-model decode layer.
//!
//! Everything here is a composition over operations the neural-net, math and
//! builder dialects already register. The layer owns the shapes a decode loop
//! needs and none of the arithmetic underneath them: a paged cache is the
//! attention layout base addressed through a block table, a sampler is the
//! mixture-of-experts top-k reduction with a draw on the end, and rotary
//! embedding at full width is the partial-rotary operation with its rotary
//! width set to the head dimension, so this layer calls that one instead of
//! declaring a second name for it.
//!
//! Consumers import from the sub-modules; this file only names them.

pub mod paged_kv;
pub mod sampling;
