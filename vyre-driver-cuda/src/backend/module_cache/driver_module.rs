//! Thin guarded wrappers over the CUDA driver module API: load, symbol lookup,
//! and unload.

use std::ffi::CStr;

use cudarc::driver::sys::{CUfunction, CUmodule, CUresult};

/// Module-scope cooperative grid-barrier counter symbol, emitted by
/// `vyre-emit-ptx` as `.global .align 4 .u32 _vyre_grid_barrier[1];`.
pub(crate) const GRID_BARRIER_SYMBOL_NAME: &str = "_vyre_grid_barrier";
pub(super) const GRID_BARRIER_SYMBOL_CSTR: &[u8] = b"_vyre_grid_barrier\0";

/// Module-scope trap record symbol, emitted by the primary text emitter as
/// `.global .align 4 .u32 _vyre_trap_sidecar[N];`. The name is owned by
/// [`vyre_emit_ptx::TRAP_SIDECAR_SYMBOL`]; the NUL-terminated form is spelled
/// here because a `CStr` cannot be built from a `&str` constant at compile time,
/// and a mismatch between the two is caught by `trap_sidecar_symbol_names_agree`.
pub(super) const TRAP_SIDECAR_SYMBOL_CSTR: &[u8] = b"_vyre_trap_sidecar\0";

pub(crate) fn load_cuda_module_data(image_with_nul: &[u8]) -> Result<CUmodule, CUresult> {
    if image_with_nul.last().copied() != Some(0) {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut module = std::ptr::null_mut();
    // SAFETY: `image_with_nul` is a live PTX/CUBIN image buffer for the call
    // duration and is checked to be NUL-terminated above.
    let result = unsafe {
        cudarc::driver::sys::cuModuleLoadData(&mut module, image_with_nul.as_ptr().cast())
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }
    if module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    Ok(module)
}

pub(crate) fn get_cuda_module_function(
    module: CUmodule,
    name: &CStr,
) -> Result<CUfunction, CUresult> {
    if module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut func = std::ptr::null_mut();
    let result = {
        // SAFETY: `module` is a CUDA module handle and `name` is a NUL-terminated
        // function symbol for the duration of the FFI call.
        unsafe { cudarc::driver::sys::cuModuleGetFunction(&mut func, module, name.as_ptr()) }
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }
    if func.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    Ok(func)
}

/// Resolve a `.global` symbol's device pointer and byte size from a loaded
/// module. Used to locate the cooperative grid-barrier counter
/// (`_vyre_grid_barrier`) that the PTX emitter declares at module scope; the
/// host zeroes it before each cooperative launch.
pub(crate) fn get_cuda_module_global(
    module: CUmodule,
    name: &CStr,
) -> Result<(u64, usize), CUresult> {
    if module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut dptr: cudarc::driver::sys::CUdeviceptr = 0;
    let mut bytes: usize = 0;
    let result = {
        // SAFETY: `module` is a loaded CUDA module handle and `name` is a
        // NUL-terminated symbol for the duration of the FFI call. `dptr`/`bytes`
        // are local out-params.
        unsafe {
            cudarc::driver::sys::cuModuleGetGlobal_v2(&mut dptr, &mut bytes, module, name.as_ptr())
        }
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }
    if dptr == 0 || bytes == 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    Ok((dptr, bytes))
}

pub(crate) fn unload_cuda_module(module: CUmodule) -> Result<(), CUresult> {
    if module.is_null() {
        return Ok(());
    }
    // SAFETY: `module` is an owned CUDA module handle; CUDA validates the
    // opaque handle and returns a CUresult.
    let result = unsafe { cudarc::driver::sys::cuModuleUnload(module) };
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(result)
    }
}

pub(super) fn unload_cuda_module_or_log(module: CUmodule, label: &str) {
    if let Err(result) = unload_cuda_module(module) {
        tracing::error!(
            "Fix: cuModuleUnload failed during {label} with {result:?}; ensure all launches using the module have completed."
        );
    }
}

// Inline: `vyre_driver_cuda::backend` is `pub(crate)`, so no integration test can reach what this
// suite exercises.
#[cfg(test)]
mod module_lifecycle_tests {
    use cudarc::driver::sys::CUresult;

    #[test]
    fn cuda_module_lifecycle_helpers_reject_invalid_handles_before_ffi() {
        let main = std::ffi::CStr::from_bytes_with_nul(b"main\0")
            .expect("Fix: test CUDA module symbol must be NUL-terminated.");
        assert_eq!(
            super::load_cuda_module_data(b".version 8.0\n").unwrap_err(),
            CUresult::CUDA_ERROR_INVALID_VALUE
        );
        assert_eq!(
            super::get_cuda_module_function(std::ptr::null_mut(), main).unwrap_err(),
            CUresult::CUDA_ERROR_INVALID_VALUE
        );
        assert!(
            super::unload_cuda_module(std::ptr::null_mut()).is_ok(),
            "Fix: null CUDA module unload should be a no-op so cleanup paths can be idempotent."
        );
    }

    /// WHY: the NUL-terminated symbol the driver looks up is a second spelling of
    /// a name the emitter owns. If the emitter renames the symbol and this copy
    /// does not follow, `cuModuleGetGlobal` returns NOT_FOUND, the sidecar is
    /// treated as absent, and a trapping launch reports success. That is the
    /// fail-open shape this whole path exists to remove, so the two spellings are
    /// pinned together rather than left to a runtime NOT_FOUND.
    ///
    /// Does not catch a rename of the array LENGTH: the word count comes from
    /// `vyre_lower::TRAP_SIDECAR_WORDS` on both sides, and a size disagreement
    /// surfaces as the byte-count check in `module_registry`.
    #[test]
    fn trap_sidecar_symbol_names_agree() {
        let with_nul = super::TRAP_SIDECAR_SYMBOL_CSTR;
        let cstr = std::ffi::CStr::from_bytes_with_nul(with_nul)
            .expect("Fix: the trap sidecar symbol literal must be NUL-terminated.");
        assert_eq!(
            cstr.to_str()
                .expect("Fix: the trap sidecar symbol must be UTF-8."),
            vyre_emit_ptx::TRAP_SIDECAR_SYMBOL,
            "Fix: the driver's NUL-terminated trap sidecar symbol drifted from the \
             emitter's. Set it to the emitter's name plus a trailing NUL."
        );
    }
}
