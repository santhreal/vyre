//! Trap recording into the module-scope trap sidecar.
//!
//! A trap is a device-side refusal: the op guards a condition the program says
//! cannot happen, and reaching it means the launch produced no valid result. The
//! host has to learn that, otherwise the launch looks successful and the garbage
//! it left in the output buffers is read as an answer. That is the shape the
//! previous lowering had: a comment and a branch to the exit label, recording
//! nothing.
//!
//! The record goes in a four-word module-scope global rather than an entry
//! parameter, for the reasons on `ModuleBuilder::write_trap_sidecar`. Word layout
//! matches the secondary text emitter exactly, so one host reader decodes either
//! target.

use std::fmt::Write as _;

use super::register_alloc::GLOBAL_ID_AXIS0_REG;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::{EmitError, TRAP_SIDECAR_SYMBOL};

/// Byte offset of the address operand within the sidecar.
const ADDRESS_WORD_OFFSET: u32 = 4;
/// Byte offset of the trap tag code within the sidecar.
const TAG_CODE_WORD_OFFSET: u32 = 8;
/// Byte offset of the reporting lane within the sidecar.
const LANE_WORD_OFFSET: u32 = 12;

impl BodyCtx<'_> {
    /// Record one trap and leave the kernel.
    ///
    /// The claim is a CAS on word 0 from 0 to 1, so among every lane that trapped
    /// in this launch exactly one writes words 1..4 and the rest fall straight
    /// through to the exit. That makes the record deterministic in content
    /// (whichever lane won reported a real trap) without needing a lock, and it
    /// means a second trap cannot overwrite the first one's operands halfway
    /// through a host read.
    ///
    /// Word 0 is claimed before the payload is written, so the record is not
    /// designed to be polled: a host that read word 0 mid-kernel could see the
    /// claim before the operands. The host reads it after a stream synchronize,
    /// which orders the kernel's writes before the readback copy, so the four
    /// words are always read together and always complete.
    pub(super) fn emit_trap(&mut self, tag: &str, address_id: u32) -> Result<(), EmitError> {
        let code = *self.trap_tag_codes.get(tag).ok_or_else(|| {
            EmitError::InvalidDescriptor(format!(
                "trap tag `{tag}` has no code in the descriptor's trap tag table, so a host \
                 reading the sidecar could not name which trap fired. Fix: emit the trap from a \
                 descriptor whose body declares it, so `vyre_lower::descriptor_trap_tags` numbers \
                 it."
            ))
        })?;
        let address = self.trap_address_operand(address_id)?;

        let sidecar = self.alloc(PtxType::U64);
        let prev = self.alloc(PtxType::U32);
        let code_reg = self.alloc(PtxType::U32);
        let first = self.alloc(PtxType::Bool);
        let recorded = self.alloc_label("trap_recorded");

        let _ = writeln!(self.text, "    // --- trap `{tag}` (code {code}) ---");
        let _ = writeln!(self.text, "    mov.u64 {sidecar}, {TRAP_SIDECAR_SYMBOL};");
        let _ = writeln!(
            self.text,
            "    atom.global.cas.b32 {prev}, [{sidecar}], 0, 1;"
        );
        let _ = writeln!(self.text, "    setp.ne.u32 {first}, {prev}, 0;");
        let _ = writeln!(self.text, "    @{first} bra {recorded};");
        let _ = writeln!(
            self.text,
            "    st.global.u32 [{sidecar} + {ADDRESS_WORD_OFFSET}], {address};"
        );
        let _ = writeln!(self.text, "    mov.u32 {code_reg}, {code};");
        let _ = writeln!(
            self.text,
            "    st.global.u32 [{sidecar} + {TAG_CODE_WORD_OFFSET}], {code_reg};"
        );
        let _ = writeln!(
            self.text,
            "    st.global.u32 [{sidecar} + {LANE_WORD_OFFSET}], {GLOBAL_ID_AXIS0_REG};"
        );
        let _ = writeln!(self.text, "{recorded}:");
        let _ = writeln!(self.text, "    bra $L_exit;");
        Ok(())
    }

    /// Coerce the trap's address operand into a `.u32` register.
    ///
    /// The sidecar word is 32 bits because the operand is an element index or
    /// byte offset into a bound buffer, not a device pointer. A `.u64` operand is
    /// truncated by `cvt`, which is correct for an offset and is what the
    /// secondary text emitter's `u32` sidecar word already does. A float or
    /// boolean operand is refused rather than bit-cast: reporting `0x3f800000` as
    /// an address tells the reader nothing and hides the descriptor defect.
    fn trap_address_operand(&mut self, address_id: u32) -> Result<Reg, EmitError> {
        let operand = self.lookup_operand(address_id)?;
        match operand.0 {
            PtxType::U32 => Ok(operand),
            PtxType::I32 => {
                let coerced = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    mov.b32 {coerced}, {operand};");
                Ok(coerced)
            }
            PtxType::U64 => {
                let coerced = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    cvt.u32.u64 {coerced}, {operand};");
                Ok(coerced)
            }
            PtxType::F32 | PtxType::B16 | PtxType::Bool => {
                Err(EmitError::InvalidDescriptor(format!(
                    "trap address operand is {}, which is not an index or offset the trap record \
                     can report. Fix: pass the integer element index the guard tested.",
                    operand.0.ptx_type_str()
                )))
            }
        }
    }
}
