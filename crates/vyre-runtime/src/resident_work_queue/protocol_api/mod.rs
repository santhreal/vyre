//! Host protocol API wrappers for megakernel control/ring buffers.

mod decode;
mod encode;
mod publish;

pub use publish::RingSlotTransition;
