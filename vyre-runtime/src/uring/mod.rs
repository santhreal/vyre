//! Linux io_uring scaffolding for NVMe → GPU-visible memory streaming.
//! All items here are gated on `cfg(target_os = "linux")` and compiled out
//! on every other platform.
//!
//! Public surface:
//! - `IoUringState`  -  raw syscall + mmap wrapper for the SQ/CQ rings.
//! - `GpuMappedBuffer`  -  typed wrapper around a GPU-visible memory region:
//!   either a registered host-visible mapping or a BAR1 peer-memory allocation.
//! - `AsyncUringStream`  -  the submission glue: pushes reads into the SQ and
//!   advances an atomic tail pointer the megakernel observes.
//! - `NvmeGpuIngestDriver`  -  publishes completed slots into the megakernel
//!   `io_queue`; `new_gpudirect` requires the native NVMe → BAR1 path.

// The submodules are the file split; the names below are the surface. Keeping
// them private leaves one public path per item instead of two.
mod driver;
mod gpudirect;
mod io_loop;
mod pump;
mod ring;
mod stream;

pub use driver::{CompletedIngest, NativeReadPath, NvmeGpuIngestDriver, NvmeGpuIngestTelemetry};
pub use gpudirect::{encode_nvme_read_sqe, GpuDirectCapability, NVME_CMD_READ};
pub use io_loop::{RegisteredIoDestination, ResidentIoLoop};
pub use pump::UringResidentQueuePump;
pub use ring::IoUringState;
pub use stream::{AsyncUringStream, GpuMappedBuffer, Iovec};
