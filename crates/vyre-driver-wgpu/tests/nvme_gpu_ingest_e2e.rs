//! Linux ingest loop smoke/e2e: file -> io_uring -> mapped slot -> live GPU.

#![cfg(feature = "device-tests")]
#![cfg(target_os = "linux")]
#![forbid(unsafe_code)]

use core::sync::atomic::AtomicU32;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use tempfile::tempdir;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_runtime::resident_work_queue::io::ResidentIoQueue;
use vyre_runtime::uring::{
    AsyncUringStream, GpuMappedBuffer, IoUringState, NativeReadPath, NvmeGpuIngestDriver,
};
use vyre_runtime::PipelineError;

const FILE_BYTES: usize = 4 * 1024 * 1024;
const HASH_WORDS: u32 = 8;

fn write_test_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("nvme-gpu-ingest.bin");
    let mut file = File::create(&path).expect("create test file");
    let pattern: Vec<u8> = (0..4096).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let mut remaining = FILE_BYTES;
    while remaining > 0 {
        let chunk = remaining.min(pattern.len());
        file.write_all(&pattern[..chunk])
            .expect("write pattern chunk");
        remaining -= chunk;
    }
    file.flush().expect("flush test file");
    file.seek(SeekFrom::Start(0)).expect("rewind test file");
    path
}

fn copy_hash_program() -> Program {
    let idx = Expr::var("idx");
    Program::wrapped(
        vec![
            BufferDecl::storage("hash_in", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(HASH_WORDS),
            BufferDecl::output("hash_out", 1, DataType::U32).with_count(HASH_WORDS),
        ],
        [HASH_WORDS, 1, 1],
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(HASH_WORDS)),
                vec![Node::Store {
                    buffer: "hash_out".into(),
                    index: idx.clone(),
                    value: Expr::load("hash_in", idx),
                }],
            ),
        ],
    )
}

#[test]
fn ingests_file_and_surfaces_hash_through_live_backend() {
    let dir = tempdir().expect("tempdir");
    let path = write_test_file(dir.path());
    let ring = match IoUringState::new(8) {
        Ok(ring) => ring,
        Err(PipelineError::IoUringSyscall { errno, .. })
            if errno == libc::EPERM || errno == libc::ENOSYS =>
        {
            panic!(
                "Fix: io_uring must be available on the local Linux host for I.3; \
                 EPERM/ENOSYS here is a configuration bug."
            );
        }
        Err(err) => panic!("unexpected driver setup failure: {err}"),
    };
    let mut target = vec![0u8; FILE_BYTES];
    let gpu_buffer = GpuMappedBuffer::from_host_visible_slice(&mut target);
    let tail = AtomicU32::new(0);
    let stream = AsyncUringStream::new(ring, gpu_buffer, &tail);
    let mut driver =
        match NvmeGpuIngestDriver::new(stream, 1, ResidentIoQueue::new(64).expect("io queue")) {
            Ok(driver) => driver,
            Err(err) => panic!("unexpected driver setup failure: {err}"),
        };
    assert_eq!(
        driver.read_path(),
        NativeReadPath::RegisteredMappedRead,
        "plain file ingest is the compatibility mapped-read path; native NVMe must use new_gpudirect + submit_native_nvme_read"
    );

    driver.submit_file(&path, 0).expect("submit ingest");
    let deadline = Instant::now() + Duration::from_secs(15);
    let completed = loop {
        let completions = driver.poll_completions().expect("poll_completions");
        if let Some(done) = completions.into_iter().next() {
            break done;
        }
        assert!(
            Instant::now() < deadline,
            "ingest completion timed out after {:?}",
            Duration::from_secs(15)
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(completed.slot, 0);
    assert_eq!(completed.byte_count as usize, FILE_BYTES);
    assert!(
        driver.megakernel_io_queue().completion(0).is_some(),
        "io_queue slot 0 must be published to the megakernel after CQE completion"
    );

    let cpu_hash = blake3::hash(
        &std::fs::read(&path)
            .expect("Fix: the ingest fixture must remain readable for CPU hashing"),
    );
    let backend = WgpuBackend::acquire()
        .expect("Fix: the live GPU backend must be available for the ingest e2e test");
    let gpu_bytes = backend
        .dispatch(
            &copy_hash_program(),
            &[cpu_hash.as_bytes().to_vec()],
            &DispatchConfig::default(),
        )
        .expect("VYRE hash-copy dispatch")
        .into_iter()
        .next()
        .expect("hash-copy output buffer");
    assert_eq!(
        &gpu_bytes[..cpu_hash.as_bytes().len()],
        cpu_hash.as_bytes(),
        "live GPU output must round-trip the ingested file's BLAKE3 digest"
    );
}

#[test]
fn gpudirect_path_fails_loudly_when_native_nvme_is_not_configured() {
    let ring = match IoUringState::new(8) {
        Ok(ring) => ring,
        Err(PipelineError::IoUringSyscall { errno, .. })
            if errno == libc::EPERM || errno == libc::ENOSYS =>
        {
            panic!(
                "Fix: io_uring must be available for the native GPUDirect ingest probe; \
                 EPERM/ENOSYS is a host configuration bug."
            );
        }
        Err(err) => panic!("unexpected ring setup failure: {err}"),
    };
    let mut target = vec![0u8; FILE_BYTES];
    let gpu_buffer = GpuMappedBuffer::from_host_visible_slice(&mut target);
    let tail = AtomicU32::new(0);
    let stream = AsyncUringStream::new(ring, gpu_buffer, &tail);
    let io_queue = match ResidentIoQueue::new(64) {
        Ok(q) => q,
        Err(err) => panic!("unexpected io queue failure: {err}"),
    };
    match NvmeGpuIngestDriver::new_gpudirect(stream, 1, io_queue) {
        Ok(driver) => assert_eq!(
            driver.read_path(),
            NativeReadPath::GpuDirectNvmePassthrough,
            "Fix: new_gpudirect must construct only the native NVMe passthrough path."
        ),
        Err(error @ PipelineError::NvmePassthroughDisabled) => {
            assert!(
                error.to_string().contains("Fix:"),
                "disabled native NVMe path must remain an explicit actionable error: {error}"
            );
        }
        Err(PipelineError::Backend(message)) => {
            assert!(
                message.contains("GPUDirect native read unavailable") && message.contains("Fix:"),
                "Fix: missing GPUDirect/nvidia-fs must be reported as an actionable native-path error, got: {message}"
            );
        }
        Err(PipelineError::IoUringSyscall { errno, .. })
            if errno == libc::EPERM || errno == libc::ENOSYS =>
        {
            panic!(
                "Fix: io_uring must be available for the native GPUDirect ingest probe; \
                 EPERM/ENOSYS is a host configuration bug."
            );
        }
        Err(error) => panic!("unexpected GPUDirect constructor failure: {error}"),
    }
}
