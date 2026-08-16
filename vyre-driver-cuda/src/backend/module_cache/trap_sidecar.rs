//! The module-scope trap record of one loaded module, and the tag table that
//! decodes it.
//!
//! The tag table is parsed out of the module's own text rather than plumbed from
//! a descriptor. A module in the cache is identified by its text and its target,
//! so a table read from that text belongs to that module by construction; a table
//! carried separately can arrive from a different descriptor and decode a real
//! trap to the wrong tag, which is worse than reporting the bare code.

use std::sync::Arc;

use vyre_driver::trap_record::TRAP_RECORD_BYTES;
use vyre_driver::BackendError;
use vyre_emit_ptx::{TRAP_SIDECAR_SYMBOL, TRAP_TAG_PTX_MARKER};

/// One loaded module's trap record and its tag table.
#[derive(Debug)]
pub(crate) struct TrapSidecar {
    /// Device pointer to word 0 of the record.
    device_ptr: u64,
    /// Bytes the module reserved for it, at least [`TRAP_RECORD_BYTES`].
    byte_count: usize,
    /// `(code, tag)` in emission order, from the module's own text.
    tags: Box<[(u32, Box<str>)]>,
}

impl TrapSidecar {
    /// Device pointer to word 0.
    pub(crate) fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Bytes to zero before a launch and read back after it.
    pub(crate) fn byte_count(&self) -> usize {
        self.byte_count
    }

    /// Tag text for a code the device recorded, or `None` for a code this module
    /// never declared.
    pub(crate) fn tag_for_code(&self, code: u32) -> Option<String> {
        self.tags
            .iter()
            .find(|(candidate, _)| *candidate == code)
            .map(|(_, tag)| tag.as_ref().to_owned())
    }
}

/// Whether `ptx_src` declares the module-scope trap record.
///
/// The emitter always names the symbol, so the text is an exact signal and no
/// speculative `cuModuleGetGlobal` is issued for the kernels that declare no
/// trap.
pub(super) fn declares_trap_sidecar(ptx_src: &str) -> bool {
    ptx_src.contains(TRAP_SIDECAR_SYMBOL)
}

/// Build the sidecar record for a module that declares one.
///
/// # Errors
///
/// Returns an error when the module reserved fewer bytes than a trap record
/// occupies, or when a tag marker in the text is malformed. Both mean the host
/// would decode a record it cannot trust, and a trap decoded wrong is reported to
/// the caller as a different failure than the one that happened.
pub(super) fn trap_sidecar_from_module(
    device_ptr: u64,
    byte_count: usize,
    ptx_src: &str,
) -> Result<Arc<TrapSidecar>, BackendError> {
    if byte_count < TRAP_RECORD_BYTES {
        return Err(BackendError::KernelCompileFailed {
            backend: crate::CUDA_BACKEND_ID.to_string(),
            compiler_message: format!(
                "loaded module reserved {byte_count} bytes for `{TRAP_SIDECAR_SYMBOL}` but a trap record is {TRAP_RECORD_BYTES} bytes. Fix: emit the symbol as vyre_lower::TRAP_SIDECAR_WORDS u32 words."
            ),
        });
    }
    Ok(Arc::new(TrapSidecar {
        device_ptr,
        byte_count,
        tags: parse_trap_tag_table(ptx_src)?,
    }))
}

/// Parse the `code tag` pairs the emitter wrote into the module text.
fn parse_trap_tag_table(ptx_src: &str) -> Result<Box<[(u32, Box<str>)]>, BackendError> {
    let mut tags: Vec<(u32, Box<str>)> = Vec::new();
    for line in ptx_src.lines() {
        let Some(rest) = line.trim_start().strip_prefix(TRAP_TAG_PTX_MARKER) else {
            continue;
        };
        let (code_text, tag) = rest.split_once(' ').ok_or_else(|| malformed(line))?;
        let code = code_text.parse::<u32>().map_err(|_| malformed(line))?;
        if code == 0 || tag.is_empty() {
            return Err(malformed(line));
        }
        if tags.iter().any(|(existing, _)| *existing == code) {
            return Err(BackendError::KernelCompileFailed {
                backend: crate::CUDA_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "loaded module declares trap tag code {code} twice, so a recorded code would decode to two tags. Fix: number trap tags through vyre_lower::descriptor_trap_tags, which assigns each distinct tag one code."
                ),
            });
        }
        tags.push((code, Box::from(tag)));
    }
    if tags.is_empty() {
        return Err(BackendError::KernelCompileFailed {
            backend: crate::CUDA_BACKEND_ID.to_string(),
            compiler_message: format!(
                "loaded module declares `{TRAP_SIDECAR_SYMBOL}` but carries no `{TRAP_TAG_PTX_MARKER}` table, so a recorded trap could only be reported as a bare code. Fix: keep the emitter's per-tag marker beside the sidecar declaration."
            ),
        });
    }
    Ok(tags.into_boxed_slice())
}

fn malformed(line: &str) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: crate::CUDA_BACKEND_ID.to_string(),
        compiler_message: format!(
            "loaded module carries a malformed trap tag marker: `{line}`. Fix: emit each marker as `{TRAP_TAG_PTX_MARKER}<nonzero code> <tag>`."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{declares_trap_sidecar, trap_sidecar_from_module};
    use vyre_driver::trap_record::TRAP_RECORD_BYTES;
    use vyre_emit_ptx::{TRAP_SIDECAR_SYMBOL, TRAP_TAG_PTX_MARKER};

    fn module_text(markers: &str) -> String {
        format!(".global .align 4 .u32 {TRAP_SIDECAR_SYMBOL}[4];\n{markers}")
    }

    /// WHY: the tag table decodes word 2 of a real device trap. A table that is
    /// silently empty, silently truncated, or silently double-booked turns a trap
    /// into a refusal that names the wrong condition, and a wrong refusal costs
    /// more to chase than a bare code. Each of these malformations therefore
    /// refuses at module load, where the module text is still in hand, instead of
    /// at readback time where only the code is.
    ///
    /// Does not catch a table whose codes are internally consistent but were
    /// numbered by something other than `vyre_lower::descriptor_trap_tags`: that
    /// is what routing both the emitter and this parser through one numbering
    /// owner is for, and no readback can detect it.
    #[test]
    fn a_module_whose_trap_table_cannot_be_trusted_refuses_at_load() {
        let good = module_text(&format!(
            "{TRAP_TAG_PTX_MARKER}1 first-tag\n{TRAP_TAG_PTX_MARKER}2 second-tag\n"
        ));
        let sidecar = trap_sidecar_from_module(0x1000, TRAP_RECORD_BYTES, &good)
            .expect("Fix: a well-formed trap tag table must load.");
        assert_eq!(sidecar.tag_for_code(1).as_deref(), Some("first-tag"));
        assert_eq!(sidecar.tag_for_code(2).as_deref(), Some("second-tag"));
        assert_eq!(sidecar.tag_for_code(3), None);
        assert_eq!(sidecar.byte_count(), TRAP_RECORD_BYTES);
        assert_eq!(sidecar.device_ptr(), 0x1000);

        for (case, text) in [
            ("no table at all", module_text("")),
            (
                "code zero",
                module_text(&format!("{TRAP_TAG_PTX_MARKER}0 zero-code\n")),
            ),
            (
                "missing tag text",
                module_text(&format!("{TRAP_TAG_PTX_MARKER}1\n")),
            ),
            (
                "non-numeric code",
                module_text(&format!("{TRAP_TAG_PTX_MARKER}one first-tag\n")),
            ),
            (
                "duplicate code",
                module_text(&format!(
                    "{TRAP_TAG_PTX_MARKER}1 first-tag\n{TRAP_TAG_PTX_MARKER}1 other-tag\n"
                )),
            ),
        ] {
            assert!(
                trap_sidecar_from_module(0x1000, TRAP_RECORD_BYTES, &text).is_err(),
                "Fix: a trap tag table with {case} must refuse at module load rather than decode a real trap to the wrong tag."
            );
        }

        assert!(
            trap_sidecar_from_module(
                0x1000,
                TRAP_RECORD_BYTES - 1,
                &module_text(&format!("{TRAP_TAG_PTX_MARKER}1 first-tag\n")),
            )
            .is_err(),
            "Fix: a sidecar smaller than one trap record must refuse at load, because the readback would decode words the module never reserved."
        );
        assert!(declares_trap_sidecar(&good));
        assert!(!declares_trap_sidecar(".visible .entry main(\n"));
    }
}
