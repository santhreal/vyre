//! Semantic persistent-BFS expansion wrappers.

mod dispatch;
mod scratch;

pub use dispatch::*;
pub use scratch::PersistentBfsGpuScratch;

#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/persistent_bfs/mod.rs"]
mod tests;
