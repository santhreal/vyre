//! Shared memory-access direction used by substrate-neutral analyses and emit patterns.

use serde::{Deserialize, Serialize};

/// Direction of a memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessKind {
    /// Read from memory.
    Load,
    /// Write to memory.
    Store,
}
