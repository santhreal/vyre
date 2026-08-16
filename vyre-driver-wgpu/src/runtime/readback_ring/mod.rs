//! Async readback ring (Innovation I.5).
//!
//! Blocking readback submits a copy + device.poll(Wait) that stalls
//! the submit queue. Under high dispatch rate this ruins latency and
//! throughput  -  the GPU goes idle while the CPU waits.
//!
//! The readback ring threads N staging buffers. Dispatch \`i\` writes
//! to \`ring[i % N]\`; the copy submits immediately and readback
//! happens asynchronously via \`map_async\`. Dispatch \`i+1\` runs in
//! parallel with readback \`i\`'s copy.

mod capacity;
mod ring;
mod ring_set;
mod slot;
mod stats;

pub use ring::ReadbackRing;
pub use ring_set::ReadbackRingSet;
pub use slot::{GpuSlot, MapResult, ReadbackTicket};
pub use stats::RingStats;
