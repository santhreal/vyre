//! Host protocol API wrappers for megakernel control/ring buffers.

mod decode;
mod encode;
mod publish;

pub(super) use decode::{validate_control_bytes, validate_debug_log_bytes};
pub use publish::RingSlotTransition;
