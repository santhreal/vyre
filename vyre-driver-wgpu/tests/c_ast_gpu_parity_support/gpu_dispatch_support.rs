use super::*;

pub(crate) fn gpu_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "WgpuBackend::acquire failed on a machine that must have a GPU. \
             This is a configuration bug, not a graceful skip.",
        )
    })
}

pub(crate) fn dispatch_gpu_program(
    context: &'static str,
    program: Program,
    inputs: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    static DISPATCH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = DISPATCH_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let input_refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        let result = gpu_backend()
            .dispatch_borrowed(&program, &input_refs, &Default::default())
            .map_err(|err| format!("{err:?}"));
        let _ = tx.send(result);
    });
    match rx.recv_timeout(Duration::from_secs(90)) {
        Ok(Ok(outputs)) => outputs,
        Ok(Err(err)) => panic!("{context}: GPU dispatch failed: {err}"),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "{context}: GPU dispatch exceeded 90s. Fix: inspect WGPU device acquisition, shader compilation, and queue completion; C parser GPU parity must fail loudly, not hang."
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{context}: GPU dispatch worker terminated before returning outputs")
        }
    }
}

/// This crate's [`ParityArm`]: the shared C-AST parity matrix dispatched on the
/// GPU.
///
/// The matrix itself, including every stage's [`Program`] and every comparison
/// against the CPU oracle, is owned by `tests/support/c_frontend/parity_matrix`
/// so the CPU-reference arm in `vyre-libs/tests` runs the same sequence. What is
/// GPU-specific and stays here is [`dispatch_gpu_program`]'s serialization and
/// watchdog.
pub(crate) struct GpuArm;

impl ParityArm for GpuArm {
    fn dispatch(
        &self,
        context: &'static str,
        program: Program,
        inputs: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        dispatch_gpu_program(context, program, inputs)
    }
}
