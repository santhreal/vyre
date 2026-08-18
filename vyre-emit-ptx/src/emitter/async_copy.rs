//! AsyncLoad / AsyncStore lowering for PTX.
//!
//! `offset` and `size` are byte counts. The reference evaluator defines an async
//! transfer as a byte span: the source reads as zero past its end, the
//! destination is clipped at its end, a load offsets the source and a store
//! offsets the destination. A binding addresses whole elements, so a span whose
//! offset or length is not a multiple of four is assembled from the two words it
//! straddles and merged into the destination under a byte mask. Shifting the
//! byte offset down to a word index instead drops the low two bits, so every
//! unaligned offset copied from the wrong byte.
//!
//! The copy runs in every thread of the launch, so a span whose ends fall inside
//! a word has several threads reading and rewriting the same destination word.
//! Each computes the same word from the same source, and the bytes a mask
//! preserves are written by no thread, so the read-modify-write is idempotent
//! however the threads interleave.
//!
//! Native `cp.async` moves four aligned bytes per instruction and cannot express
//! a byte-granular span, so it stays reserved for a statically word-aligned
//! offset and the general path carries the rest.

use std::fmt::Write as _;

use vyre_foundation::ir::DataType;
use vyre_lower::MemoryClass;
use vyre_lower::Name;

use super::memory::AsyncCopyDirection;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

/// A byte offset resolved onto the word grid of a four-byte binding.
struct ByteSpan {
    /// First word the span touches.
    word_start: Reg,
    /// Byte position of the span's first byte inside `word_start`.
    byte_in_word: Reg,
    /// Bit position of the span's first byte inside `word_start`.
    bit_shift: Reg,
    /// Set when the offset is a known multiple of four, so the span starts on a
    /// word boundary and needs no cross-word assembly.
    word_aligned: bool,
}

impl BodyCtx<'_> {
    pub(super) fn emit_u32_const(&mut self, value: u32) -> Reg {
        let reg = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    mov.u32    {reg}, {value};");
        reg
    }

    fn emit_words_from_byte_size(&mut self, size_reg: Reg) -> Reg {
        let rounded = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    add.u32    {rounded}, {size_reg}, 3;");
        let words = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    shr.u32    {words}, {rounded}, 2;");
        words
    }

    fn emit_min_u32(&mut self, left: Reg, right: Reg) -> Reg {
        let pred = self.alloc(PtxType::Bool);
        let out = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    setp.lt.u32    {pred}, {left}, {right};");
        let _ = writeln!(self.text, "    selp.u32    {out}, {left}, {right}, {pred};");
        out
    }

    fn emit_binding_len_or_max(&mut self, slot: u32) -> Result<Reg, EmitError> {
        let count = self
            .binding_for_slot(slot)?
            .element_count
            .unwrap_or(u32::MAX);
        Ok(self.emit_u32_const(count))
    }

    /// One `.u32` instruction with two register operands.
    fn emit_u32_op(&mut self, mnemonic: &str, left: Reg, right: Reg) -> Reg {
        let out = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    {mnemonic}    {out}, {left}, {right};");
        out
    }

    fn emit_add_u32(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_u32_op("add.u32   ", left, right)
    }

    fn emit_sub_u32(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_u32_op("sub.u32   ", left, right)
    }

    fn emit_shl_u32(&mut self, value: Reg, bits: Reg) -> Reg {
        self.emit_u32_op("shl.b32   ", value, bits)
    }

    fn emit_shr_u32(&mut self, value: Reg, bits: Reg) -> Reg {
        self.emit_u32_op("shr.u32   ", value, bits)
    }

    fn emit_and_u32(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_u32_op("and.b32   ", left, right)
    }

    fn emit_or_u32(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_u32_op("or.b32    ", left, right)
    }

    fn emit_not_u32(&mut self, value: Reg) -> Reg {
        let out = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    not.b32    {out}, {value};");
        out
    }

    fn emit_lt_u32(&mut self, left: Reg, right: Reg) -> Reg {
        let pred = self.alloc(PtxType::Bool);
        let _ = writeln!(self.text, "    setp.lt.u32    {pred}, {left}, {right};");
        pred
    }

    fn emit_select_u32(&mut self, condition: Reg, accept: Reg, reject: Reg) -> Reg {
        let out = self.alloc(PtxType::U32);
        let _ = writeln!(
            self.text,
            "    selp.u32    {out}, {accept}, {reject}, {condition};"
        );
        out
    }

    /// Resolve a byte offset onto the word grid.
    fn emit_byte_span(&mut self, offset: Reg, offset_id: u32) -> ByteSpan {
        let two = self.emit_u32_const(2);
        let word_start = self.emit_shr_u32(offset, two);
        let word_aligned = self
            .u32_literals
            .get(&offset_id)
            .is_some_and(|bytes| bytes % 4 == 0);
        if word_aligned {
            let zero = self.emit_u32_const(0);
            return ByteSpan {
                word_start,
                byte_in_word: zero,
                bit_shift: zero,
                word_aligned,
            };
        }
        let three = self.emit_u32_const(3);
        let byte_in_word = self.emit_and_u32(offset, three);
        let bit_shift = self.emit_shl_u32(byte_in_word, three);
        ByteSpan {
            word_start,
            byte_in_word,
            bit_shift,
            word_aligned,
        }
    }

    /// Whether a transfer length is a known whole number of words.
    fn byte_size_is_whole_words(&self, size_id: u32) -> bool {
        self.u32_literals
            .get(&size_id)
            .is_some_and(|bytes| bytes % 4 == 0)
    }

    /// A binding whose elements are not four bytes wide has no word grid for a
    /// byte span to sit on.
    fn require_word_element(&self, slot: u32, label: &str) -> Result<DataType, EmitError> {
        let element = self.binding_for_slot(slot)?.element_type.clone();
        let width = element.size_bytes();
        if width == Some(4) {
            return Ok(element);
        }
        Err(EmitError::UnsupportedDataType(format!(
            "async copy {label} element {element:?} is not four bytes wide. Fix: stage the transfer through a U32 buffer, whose word grid a byte span can address."
        )))
    }

    /// The word of `slot` at `index` as a bit pattern, or zero when `index` is
    /// past the end.
    ///
    /// The load is `.u32` whatever the declared element type: a transfer copies
    /// bits, and a float that never enters an arithmetic instruction needs no
    /// float register.
    fn emit_word_or_zero(
        &mut self,
        slot: u32,
        index: Reg,
        len: Reg,
        element: &DataType,
        class: MemoryClass,
    ) -> Result<Reg, EmitError> {
        let in_bounds = self.emit_lt_u32(index, len);
        let zero = self.emit_u32_const(0);
        let safe_index = self.emit_select_u32(in_bounds, index, zero);
        let address = self.emit_memory_address_from_index_reg(slot, safe_index, element, class)?;
        let value = self.alloc(PtxType::U32);
        let space = self.load_space_for(slot, class);
        let _ = write!(self.text, "    @{in_bounds} ld.{space}.u32    {value}, ");
        self.write_mem_operand(address.operand)?;
        self.text.push_str(";\n");
        let _ = writeln!(self.text, "    @!{in_bounds} mov.u32    {value}, 0;");
        Ok(value)
    }

    /// The four span bytes that begin `bit_shift` bits into word `index`.
    fn emit_span_word(
        &mut self,
        slot: u32,
        index: Reg,
        len: Reg,
        element: &DataType,
        class: MemoryClass,
        span: &ByteSpan,
    ) -> Result<Reg, EmitError> {
        let low_word = self.emit_word_or_zero(slot, index, len, element, class)?;
        if span.word_aligned {
            return Ok(low_word);
        }
        let one = self.emit_u32_const(1);
        let next_index = self.emit_add_u32(index, one);
        let high_word = self.emit_word_or_zero(slot, next_index, len, element, class)?;
        let low_part = self.emit_shr_u32(low_word, span.bit_shift);
        let high_part = self.emit_shifted_carry(high_word, span.bit_shift, true);
        Ok(self.emit_or_u32(low_part, high_part))
    }

    /// The payload bytes that land in the destination word at `index`.
    ///
    /// A store shifts the payload up by the destination's byte offset, so each
    /// destination word takes the low bytes of one payload word and the high
    /// bytes of the one before it. The predecessor index wraps below zero on the
    /// first word, which reads as out of bounds and contributes nothing.
    fn emit_payload_word(
        &mut self,
        slot: u32,
        index: Reg,
        len: Reg,
        element: &DataType,
        class: MemoryClass,
        span: &ByteSpan,
    ) -> Result<Reg, EmitError> {
        let current = self.emit_word_or_zero(slot, index, len, element, class)?;
        if span.word_aligned {
            return Ok(current);
        }
        let one = self.emit_u32_const(1);
        let previous_index = self.emit_sub_u32(index, one);
        let previous = self.emit_word_or_zero(slot, previous_index, len, element, class)?;
        let low_part = self.emit_shl_u32(current, span.bit_shift);
        let high_part = self.emit_shifted_carry(previous, span.bit_shift, false);
        Ok(self.emit_or_u32(low_part, high_part))
    }

    /// The part of an adjacent word that a shift of `bit_shift` carries in.
    ///
    /// The complementary shift is `32 - bit_shift`, which is out of range when
    /// `bit_shift` is zero, so the amount is masked to five bits and the zero
    /// case selects the carry away.
    fn emit_shifted_carry(&mut self, word: Reg, bit_shift: Reg, toward_high: bool) -> Reg {
        let thirty_two = self.emit_u32_const(32);
        let complement = self.emit_sub_u32(thirty_two, bit_shift);
        let thirty_one = self.emit_u32_const(31);
        let amount = self.emit_and_u32(complement, thirty_one);
        let carry = if toward_high {
            self.emit_shl_u32(word, amount)
        } else {
            self.emit_shr_u32(word, amount)
        };
        let zero = self.emit_u32_const(0);
        let shifted = self.emit_lt_u32(zero, bit_shift);
        self.emit_select_u32(shifted, carry, zero)
    }

    /// `value` with its low `bytes` bytes cleared. `bytes` must be 0..=3.
    fn emit_byte_mask_above(&mut self, value: Reg, bytes: Reg) -> Reg {
        let three = self.emit_u32_const(3);
        let bits = self.emit_shl_u32(bytes, three);
        self.emit_shl_u32(value, bits)
    }

    /// `value` with its high `bytes` bytes cleared. `bytes` must be 0..=3.
    fn emit_byte_mask_below(&mut self, value: Reg, bytes: Reg) -> Reg {
        let three = self.emit_u32_const(3);
        let bits = self.emit_shl_u32(bytes, three);
        self.emit_shr_u32(value, bits)
    }

    /// The destination bytes a load must preserve in the word at `index`.
    fn emit_load_tail_keep_mask(&mut self, size: Reg, index: Reg) -> Reg {
        let two = self.emit_u32_const(2);
        let four = self.emit_u32_const(4);
        let consumed = self.emit_shl_u32(index, two);
        let remaining = self.emit_sub_u32(size, consumed);
        let carried = self.emit_min_u32(remaining, four);
        let partial = self.emit_lt_u32(carried, four);
        let all_ones = self.emit_u32_const(u32::MAX);
        let mask = self.emit_byte_mask_above(all_ones, carried);
        let zero = self.emit_u32_const(0);
        self.emit_select_u32(partial, mask, zero)
    }

    /// The destination bytes a store writes in the word at `index`.
    fn emit_store_write_mask(&mut self, size: Reg, index: Reg, span: &ByteSpan) -> Reg {
        let one = self.emit_u32_const(1);
        let two = self.emit_u32_const(2);
        let four = self.emit_u32_const(4);
        let zero = self.emit_u32_const(0);
        let all_ones = self.emit_u32_const(u32::MAX);

        let first_word = self.emit_lt_u32(index, one);
        let skipped = self.emit_select_u32(first_word, span.byte_in_word, zero);
        let below = self.emit_byte_mask_above(all_ones, skipped);

        let consumed = self.emit_shl_u32(index, two);
        let limit = self.emit_add_u32(size, span.byte_in_word);
        let remaining = self.emit_sub_u32(limit, consumed);
        let written = self.emit_min_u32(remaining, four);
        let unwritten = self.emit_sub_u32(four, written);
        let above = self.emit_byte_mask_below(all_ones, unwritten);

        self.emit_and_u32(below, above)
    }

    /// `value` with the bytes `keep` selects taken from `slot[index]` instead.
    fn emit_merge_under_mask(
        &mut self,
        slot: u32,
        index: Reg,
        element: &DataType,
        class: MemoryClass,
        value: Reg,
        keep: Reg,
    ) -> Result<Reg, EmitError> {
        let address = self.emit_memory_address_from_index_reg(slot, index, element, class)?;
        let existing = self.alloc(PtxType::U32);
        let space = address.space;
        let _ = write!(self.text, "    ld.{space}.u32    {existing}, ");
        self.write_mem_operand(address.operand)?;
        self.text.push_str(";\n");
        let overwrite = self.emit_not_u32(keep);
        let fresh = self.emit_and_u32(value, overwrite);
        let preserved = self.emit_and_u32(existing, keep);
        Ok(self.emit_or_u32(fresh, preserved))
    }

    pub(super) fn emit_async_copy_loop(
        &mut self,
        tag: &str,
        source_slot: u32,
        destination_slot: u32,
        offset_id: u32,
        size_id: u32,
        direction: AsyncCopyDirection,
    ) -> Result<(), EmitError> {
        let source_element = self.require_word_element(source_slot, "source")?;
        let destination_element = self.require_word_element(destination_slot, "destination")?;
        let source_class = self.binding_for_slot(source_slot)?.memory_class;
        let destination_class = self.binding_for_slot(destination_slot)?.memory_class;
        let offset_reg = self.lookup_operand(offset_id)?;
        let size_reg = self.lookup_operand(size_id)?;
        let span = self.emit_byte_span(offset_reg, offset_id);
        let whole_words = self.byte_size_is_whole_words(size_id);
        let source_len = self.emit_binding_len_or_max(source_slot)?;
        let destination_len = self.emit_binding_len_or_max(destination_slot)?;
        let zero = self.emit_u32_const(0);

        // A load fills the destination from its start, so it copies as many
        // words as the destination holds. A store lands at the offset, so it
        // copies what fits after it, counting the partial first word the offset
        // starts inside. Neither shortens the copy to the source length: a span
        // that runs off the source reads zeros.
        let copy_words = match direction {
            AsyncCopyDirection::Load => {
                let requested_words = self.emit_words_from_byte_size(size_reg);
                self.emit_min_u32(requested_words, destination_len)
            }
            AsyncCopyDirection::Store => {
                let spanned_bytes = self.emit_add_u32(size_reg, span.byte_in_word);
                let spanned_words = self.emit_words_from_byte_size(spanned_bytes);
                let nonempty = self.emit_lt_u32(zero, size_reg);
                let spanned_words = self.emit_select_u32(nonempty, spanned_words, zero);
                let has_space = self.emit_lt_u32(span.word_start, destination_len);
                let remaining = self.emit_sub_u32(destination_len, span.word_start);
                let destination_remaining = self.emit_select_u32(has_space, remaining, zero);
                self.emit_min_u32(spanned_words, destination_remaining)
            }
        };
        let masked = !(span.word_aligned && whole_words);

        let loop_index = self.emit_u32_const(0);
        let loop_label = self.alloc_label("async_copy");
        let done_label = self.alloc_label("async_done");
        let _ = writeln!(self.text, "{loop_label}:");
        let done_pred = self.alloc(PtxType::Bool);
        let _ = writeln!(
            self.text,
            "    setp.ge.u32    {done_pred}, {loop_index}, {copy_words};"
        );
        let _ = writeln!(self.text, "    @{done_pred} bra    {done_label};");

        let (value, destination_index) = match direction {
            AsyncCopyDirection::Load => {
                let source_index = self.emit_add_u32(span.word_start, loop_index);
                let fetched = self.emit_span_word(
                    source_slot,
                    source_index,
                    source_len,
                    &source_element,
                    source_class,
                    &span,
                )?;
                let value = if masked {
                    let keep = self.emit_load_tail_keep_mask(size_reg, loop_index);
                    self.emit_merge_under_mask(
                        destination_slot,
                        loop_index,
                        &destination_element,
                        destination_class,
                        fetched,
                        keep,
                    )?
                } else {
                    fetched
                };
                (value, loop_index)
            }
            AsyncCopyDirection::Store => {
                let destination_index = self.emit_add_u32(span.word_start, loop_index);
                let payload = self.emit_payload_word(
                    source_slot,
                    loop_index,
                    source_len,
                    &source_element,
                    source_class,
                    &span,
                )?;
                let value = if masked {
                    let written = self.emit_store_write_mask(size_reg, loop_index, &span);
                    let keep = self.emit_not_u32(written);
                    self.emit_merge_under_mask(
                        destination_slot,
                        destination_index,
                        &destination_element,
                        destination_class,
                        payload,
                        keep,
                    )?
                } else {
                    payload
                };
                (value, destination_index)
            }
        };

        let destination_addr = self.emit_memory_address_from_index_reg(
            destination_slot,
            destination_index,
            &destination_element,
            destination_class,
        )?;
        let _ = write!(self.text, "    st.{}.u32    ", destination_addr.space);
        self.write_mem_operand(destination_addr.operand)?;
        let _ = writeln!(self.text, ", {value};");
        let _ = writeln!(self.text, "    add.u32    {loop_index}, {loop_index}, 1;");
        let _ = writeln!(self.text, "    bra    {loop_label};");
        let _ = writeln!(self.text, "{done_label}:");
        let _ = writeln!(
            self.text,
            "    // async_copy tag={tag} lowered as bounded synchronous copy"
        );
        Ok(())
    }

    pub(super) fn emit_cp_async_load_loop(
        &mut self,
        tag: &Name,
        source_slot: u32,
        destination_slot: u32,
        offset_id: u32,
        size_id: u32,
    ) -> Result<bool, EmitError> {
        if !self.options.target.supports_async_copy() {
            return Ok(false);
        }
        // `cp.async` moves four naturally-aligned bytes per instruction, so it
        // can only serve a span whose offset and size start and end on word
        // boundaries. A transfer whose alignment is unknown at emit time goes
        // to the general path, which assembles each word from the two the span
        // straddles and merges partial tail bytes.
        let offset_aligned = self
            .u32_literals
            .get(&offset_id)
            .is_some_and(|bytes| bytes % 4 == 0);
        let size_aligned = self
            .u32_literals
            .get(&size_id)
            .is_some_and(|bytes| bytes % 4 == 0);
        if !(offset_aligned && size_aligned) {
            return Ok(false);
        }
        let (source_type, source_class) =
            match self.require_u32_slot(source_slot, "cp.async source") {
                Ok(v) => v,
                Err(_) => return Ok(false),
            };
        let (destination_type, destination_class) =
            match self.require_u32_slot(destination_slot, "cp.async destination") {
                Ok(v) => v,
                Err(_) => return Ok(false),
            };
        if source_class != MemoryClass::Global || destination_class != MemoryClass::Shared {
            return Ok(false);
        }

        let offset_reg = self.lookup_operand(offset_id)?;
        let size_reg = self.lookup_operand(size_id)?;
        let requested_words = self.emit_words_from_byte_size(size_reg);
        let destination_len = self.emit_binding_len_or_max(destination_slot)?;
        let source_len = self.emit_binding_len_or_max(source_slot)?;
        let two = self.emit_u32_const(2);
        let offset_words = self.emit_shr_u32(offset_reg, two);
        let copy_words = self.emit_min_u32(requested_words, destination_len);
        let zero = self.emit_u32_const(0);
        let loop_index = self.emit_u32_const(0);
        let loop_label = self.alloc_label("cp_async");
        let done_label = self.alloc_label("cp_async_done");

        let _ = writeln!(
            self.text,
            "    // cp.async_load tag={tag} src=slot{source_slot} dst=slot{destination_slot}"
        );
        let _ = writeln!(self.text, "{loop_label}:");
        let done_pred = self.alloc(PtxType::Bool);
        let _ = writeln!(
            self.text,
            "    setp.ge.u32    {done_pred}, {loop_index}, {copy_words};"
        );
        let _ = writeln!(self.text, "    @{done_pred} bra    {done_label};");

        let source_index = self.alloc(PtxType::U32);
        let _ = writeln!(
            self.text,
            "    add.u32    {source_index}, {offset_words}, {loop_index};"
        );
        let destination_index = loop_index;
        let source_addr = self.emit_memory_address_from_index_reg(
            source_slot,
            source_index,
            &source_type,
            source_class,
        )?;
        let destination_addr = self.emit_memory_address_from_index_reg(
            destination_slot,
            destination_index,
            &destination_type,
            destination_class,
        )?;
        let in_bounds = self.alloc(PtxType::Bool);
        let _ = writeln!(
            self.text,
            "    setp.lt.u32    {in_bounds}, {source_index}, {source_len};"
        );
        let _ = write!(self.text, "    @{in_bounds} cp.async.ca.shared.global    ");
        self.write_mem_operand(destination_addr.operand)?;
        self.text.push_str(", ");
        self.write_mem_operand(source_addr.operand)?;
        self.text.push_str(", 4;\n");
        let _ = write!(self.text, "    @!{in_bounds} st.shared.u32    ");
        self.write_mem_operand(destination_addr.operand)?;
        let _ = writeln!(self.text, ", {zero};");
        let _ = writeln!(self.text, "    add.u32    {loop_index}, {loop_index}, 1;");
        let _ = writeln!(self.text, "    bra    {loop_label};");
        let _ = writeln!(self.text, "{done_label}:");
        let _ = writeln!(self.text, "    cp.async.commit_group;");
        self.pending_cp_async_tags.insert(tag.clone());
        Ok(true)
    }

    pub(super) fn emit_cp_async_wait_for_tag(&mut self, tag: &str) -> bool {
        if !self.pending_cp_async_tags.remove(tag) {
            return false;
        }
        let _ = writeln!(self.text, "    cp.async.wait_group 0;");
        let _ = writeln!(self.text, "    membar.cta;");
        true
    }
}
