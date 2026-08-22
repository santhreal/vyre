//! AsyncLoad / AsyncStore op emitters. Both lower to a counted copy loop over
//! the destination words the transfer touches.
//!
//! `offset` and `size` are byte counts. The reference evaluator defines an async
//! transfer as a byte span (`Buffer::read_window`): the source is zero past its
//! end, the destination is clipped at its end, and a load applies the offset to
//! the source while a store applies it to the destination. A u32 storage binding
//! addresses words, so a span whose offset or length is not a multiple of four
//! is assembled from two adjacent words and merged into the destination under a
//! byte mask. Dividing the byte offset down to a word index instead drops the
//! low two bits and copies from the wrong byte.
//!
//! A masked merge reads the destination word it is about to write. Every
//! invocation runs the same loop over the same source, so every writer computes
//! the same word, and the bytes a mask preserves lie outside the span and are
//! written by no invocation. The read observes the same bytes whether or not
//! another invocation already stored that word.
//!
//! An offset and size that are literal multiples of four take neither path: the
//! loop stays one load and one store per word.

use naga::{BinaryOperator, Block, Expression, Span, Statement};
use vyre_lower::KernelOp;

use super::BodyBuilder;
use crate::EmitError;

/// A byte offset resolved onto the word grid of a u32 storage binding.
struct ByteSpan {
    /// First word the span touches.
    word_start: naga::Handle<Expression>,
    /// Byte position of the span's first byte inside `word_start`.
    byte_in_word: naga::Handle<Expression>,
    /// Bit position of the span's first byte inside `word_start`.
    bit_shift: naga::Handle<Expression>,
    /// Set when the offset is a literal multiple of four, so the span starts on
    /// a word boundary and needs no cross-word assembly.
    word_aligned: bool,
}

impl BodyBuilder<'_> {
    fn is_leader_invocation(&mut self) -> Result<naga::Handle<Expression>, EmitError> {
        let local_arg = self.append_expr(Expression::FunctionArgument(self.builtins.local));
        let lx = self.append_expr(Expression::AccessIndex {
            base: local_arg,
            index: 0,
        });
        let ly = self.append_expr(Expression::AccessIndex {
            base: local_arg,
            index: 1,
        });
        let lz = self.append_expr(Expression::AccessIndex {
            base: local_arg,
            index: 2,
        });
        let lxy = self.or_u32(lx, ly);
        let lxyz = self.or_u32(lxy, lz);
        let zero = self.literal_u32(0);
        Ok(self.append_expr(Expression::Binary {
            op: BinaryOperator::Equal,
            left: lxyz,
            right: zero,
        }))
    }

    pub(super) fn emit_async_load(&mut self, op: &KernelOp) -> Result<(), EmitError> {
        let source_slot = self.slot_operand(op, 0)?;
        let destination_slot = self.slot_operand(op, 1)?;
        self.require_u32_slot(source_slot, "AsyncLoad source")?;
        self.require_u32_slot(destination_slot, "AsyncLoad destination")?;

        let offset = self.value_operand(op, 2)?;
        let size = self.value_operand(op, 3)?;
        let span = self.byte_span(offset);
        let whole_words = self.is_whole_words(size);
        let requested_words = self.byte_size_to_words(size);
        let destination_len = self.buffer_len_expr(destination_slot)?;
        let source_len = self.buffer_len_expr(source_slot)?;
        let copy_words = self.min_u32(requested_words, destination_len);
        let is_leader = self.is_leader_invocation()?;

        let outer = std::mem::replace(&mut self.function.body, Block::new());
        self.emit_counted_u32_loop("async_load_word", copy_words, |this, index| {
            let source_index = this.add_u32(span.word_start, index);
            let fetched = this.span_word(source_slot, source_index, source_len, &span)?;
            let value = if whole_words {
                fetched
            } else {
                let keep = this.load_tail_keep_mask(size, index);
                this.merge_under_mask(destination_slot, index, fetched, keep)?
            };
            let destination_pointer =
                this.binding_element_pointer_by_slot(destination_slot, index)?;
            this.function.body.push(
                Statement::Store {
                    pointer: destination_pointer,
                    value,
                },
                Span::UNDEFINED,
            );
            Ok(())
        })?;
        let leader_block = std::mem::replace(&mut self.function.body, outer);
        self.function.body.push(
            Statement::If {
                condition: is_leader,
                accept: leader_block,
                reject: Block::new(),
            },
            Span::UNDEFINED,
        );
        Ok(())
    }

    pub(super) fn emit_async_store(&mut self, op: &KernelOp) -> Result<(), EmitError> {
        let source_slot = self.slot_operand(op, 0)?;
        let destination_slot = self.slot_operand(op, 1)?;
        self.require_u32_slot(source_slot, "AsyncStore source")?;
        self.require_u32_slot(destination_slot, "AsyncStore destination")?;

        let offset = self.value_operand(op, 2)?;
        let size = self.value_operand(op, 3)?;
        let span = self.byte_span(offset);
        let whole_words = self.is_whole_words(size);
        let source_len = self.buffer_len_expr(source_slot)?;
        let destination_len = self.buffer_len_expr(destination_slot)?;

        // The span occupies `size` bytes starting `byte_in_word` bytes into the
        // first word, so it touches one more word than the size alone implies
        // whenever the two do not add up to a whole number of words.
        let spanned_bytes = self.add_u32(size, span.byte_in_word);
        let spanned_words = self.byte_size_to_words(spanned_bytes);
        let zero = self.literal_u32(0);
        let nonempty = self.lt_u32(zero, size);
        let spanned_words = self.append_expr(Expression::Select {
            condition: nonempty,
            accept: spanned_words,
            reject: zero,
        });
        let destination_remaining = {
            let has_space = self.lt_u32(span.word_start, destination_len);
            let remaining = self.sub_u32(destination_len, span.word_start);
            self.append_expr(Expression::Select {
                condition: has_space,
                accept: remaining,
                reject: zero,
            })
        };
        let copy_words = self.min_u32(spanned_words, destination_remaining);
        let masked = !(span.word_aligned && whole_words);

        let is_leader = self.is_leader_invocation()?;
        let outer = std::mem::replace(&mut self.function.body, Block::new());
        self.emit_counted_u32_loop("async_store_word", copy_words, |this, index| {
            let destination_index = this.add_u32(span.word_start, index);
            let payload = this.payload_word(source_slot, index, source_len, &span)?;
            let value = if masked {
                let written = this.store_write_mask(size, index, &span);
                let keep = this.not_u32(written);
                this.merge_under_mask(destination_slot, destination_index, payload, keep)?
            } else {
                payload
            };
            let destination_pointer =
                this.binding_element_pointer_by_slot(destination_slot, destination_index)?;
            this.function.body.push(
                Statement::Store {
                    pointer: destination_pointer,
                    value,
                },
                Span::UNDEFINED,
            );
            Ok(())
        })?;
        let leader_block = std::mem::replace(&mut self.function.body, outer);
        self.function.body.push(
            Statement::If {
                condition: is_leader,
                accept: leader_block,
                reject: Block::new(),
            },
            Span::UNDEFINED,
        );
        Ok(())
    }

    /// Resolve a byte offset onto the word grid.
    ///
    /// The element size is a literal four, so the division and remainder are a
    /// shift and a mask: the NVIDIA shader compiler emits an integer divide for
    /// the general form even when the divisor is a constant.
    fn byte_span(&mut self, offset: naga::Handle<Expression>) -> ByteSpan {
        let word_aligned = self
            .literal_u32_of(offset)
            .is_some_and(|byte| byte % 4 == 0);
        let two = self.literal_u32(2);
        let word_start = self.shr_u32(offset, two);
        if word_aligned {
            let zero = self.literal_u32(0);
            return ByteSpan {
                word_start,
                byte_in_word: zero,
                bit_shift: zero,
                word_aligned,
            };
        }
        let three = self.literal_u32(3);
        let byte_in_word = self.and_u32(offset, three);
        let bit_shift = self.shl_u32(byte_in_word, three);
        ByteSpan {
            word_start,
            byte_in_word,
            bit_shift,
            word_aligned,
        }
    }

    /// Whether the transfer length is a literal whole number of words.
    fn is_whole_words(&self, size: naga::Handle<Expression>) -> bool {
        self.literal_u32_of(size)
            .is_some_and(|bytes| bytes % 4 == 0)
    }

    /// The word of `slot` at `index`, or zero when `index` is past the end.
    ///
    /// Zero past the end is the reference's `read_window` padding: a span that
    /// runs off the source reads zeros rather than shortening the transfer.
    fn word_or_zero(
        &mut self,
        slot: u32,
        index: naga::Handle<Expression>,
        len: naga::Handle<Expression>,
    ) -> Result<naga::Handle<Expression>, EmitError> {
        let in_bounds = self.lt_u32(index, len);
        let zero = self.literal_u32(0);
        let safe_index = self.append_expr(Expression::Select {
            condition: in_bounds,
            accept: index,
            reject: zero,
        });
        let pointer = self.binding_element_pointer_by_slot(slot, safe_index)?;
        let loaded = self.append_expr(Expression::Load { pointer });
        Ok(self.append_expr(Expression::Select {
            condition: in_bounds,
            accept: loaded,
            reject: zero,
        }))
    }

    /// The four span bytes that begin `bit_shift` bits into word `index`.
    fn span_word(
        &mut self,
        slot: u32,
        index: naga::Handle<Expression>,
        len: naga::Handle<Expression>,
        span: &ByteSpan,
    ) -> Result<naga::Handle<Expression>, EmitError> {
        let low_word = self.word_or_zero(slot, index, len)?;
        if span.word_aligned {
            return Ok(low_word);
        }
        let one = self.literal_u32(1);
        let next_index = self.add_u32(index, one);
        let high_word = self.word_or_zero(slot, next_index, len)?;
        let low_part = self.shr_u32(low_word, span.bit_shift);
        let high_part = self.shifted_carry(high_word, span.bit_shift, true);
        Ok(self.or_u32(low_part, high_part))
    }

    /// The payload bytes that land in the destination word at `index`.
    ///
    /// A store shifts the payload up by the destination's byte offset, so each
    /// destination word takes the low bytes of one payload word and the high
    /// bytes of the one before it. The predecessor index wraps below zero on the
    /// first word, which reads as out of bounds and contributes nothing.
    fn payload_word(
        &mut self,
        slot: u32,
        index: naga::Handle<Expression>,
        len: naga::Handle<Expression>,
        span: &ByteSpan,
    ) -> Result<naga::Handle<Expression>, EmitError> {
        let current = self.word_or_zero(slot, index, len)?;
        if span.word_aligned {
            return Ok(current);
        }
        let one = self.literal_u32(1);
        let previous_index = self.sub_u32(index, one);
        let previous = self.word_or_zero(slot, previous_index, len)?;
        let low_part = self.shl_u32(current, span.bit_shift);
        let high_part = self.shifted_carry(previous, span.bit_shift, false);
        Ok(self.or_u32(low_part, high_part))
    }

    /// The part of an adjacent word that a shift of `bit_shift` carries in.
    ///
    /// The complementary shift is `32 - bit_shift`, which is an out-of-range
    /// shift amount when `bit_shift` is zero, so the amount is masked to five
    /// bits and the zero case selects the carry away. Both arms of a select are
    /// evaluated, so the masked amount has to stay in range on the arm the
    /// select discards.
    fn shifted_carry(
        &mut self,
        word: naga::Handle<Expression>,
        bit_shift: naga::Handle<Expression>,
        toward_high: bool,
    ) -> naga::Handle<Expression> {
        let thirty_two = self.literal_u32(32);
        let complement = self.sub_u32(thirty_two, bit_shift);
        let thirty_one = self.literal_u32(31);
        let amount = self.and_u32(complement, thirty_one);
        let carry = if toward_high {
            self.shl_u32(word, amount)
        } else {
            self.shr_u32(word, amount)
        };
        let zero = self.literal_u32(0);
        let shifted = self.lt_u32(zero, bit_shift);
        self.append_expr(Expression::Select {
            condition: shifted,
            accept: carry,
            reject: zero,
        })
    }

    /// The destination bytes a load must preserve in the word at `index`.
    ///
    /// A transfer length that is not a whole number of words ends inside a word.
    /// The bytes above the end belong to the destination and stay as they were.
    fn load_tail_keep_mask(
        &mut self,
        size: naga::Handle<Expression>,
        index: naga::Handle<Expression>,
    ) -> naga::Handle<Expression> {
        let two = self.literal_u32(2);
        let four = self.literal_u32(4);
        let consumed = self.shl_u32(index, two);
        let remaining = self.sub_u32(size, consumed);
        let carried = self.min_u32(remaining, four);
        let partial = self.lt_u32(carried, four);
        let all_ones = self.literal_u32(u32::MAX);
        let mask = self.byte_mask_above(all_ones, carried);
        let zero = self.literal_u32(0);
        self.append_expr(Expression::Select {
            condition: partial,
            accept: mask,
            reject: zero,
        })
    }

    /// The destination bytes a store writes in the word at `index`.
    ///
    /// The span starts `byte_in_word` bytes into the first word and ends
    /// `size + byte_in_word` bytes later, so the first and last words it touches
    /// are partial. Bytes outside that range belong to the destination.
    fn store_write_mask(
        &mut self,
        size: naga::Handle<Expression>,
        index: naga::Handle<Expression>,
        span: &ByteSpan,
    ) -> naga::Handle<Expression> {
        let zero = self.literal_u32(0);
        let one = self.literal_u32(1);
        let two = self.literal_u32(2);
        let four = self.literal_u32(4);
        let all_ones = self.literal_u32(u32::MAX);

        let first_word = self.lt_u32(index, one);
        let skipped = self.append_expr(Expression::Select {
            condition: first_word,
            accept: span.byte_in_word,
            reject: zero,
        });
        let below = self.byte_mask_above(all_ones, skipped);

        let consumed = self.shl_u32(index, two);
        let limit = self.add_u32(size, span.byte_in_word);
        let remaining = self.sub_u32(limit, consumed);
        let written = self.min_u32(remaining, four);
        let unwritten = self.sub_u32(four, written);
        let above = self.byte_mask_below(all_ones, unwritten);

        self.and_u32(below, above)
    }

    /// `value` with its low `bytes` bytes cleared. `bytes` must be 0..=3.
    fn byte_mask_above(
        &mut self,
        value: naga::Handle<Expression>,
        bytes: naga::Handle<Expression>,
    ) -> naga::Handle<Expression> {
        let three = self.literal_u32(3);
        let bits = self.shl_u32(bytes, three);
        self.shl_u32(value, bits)
    }

    /// `value` with its high `bytes` bytes cleared. `bytes` must be 0..=3.
    fn byte_mask_below(
        &mut self,
        value: naga::Handle<Expression>,
        bytes: naga::Handle<Expression>,
    ) -> naga::Handle<Expression> {
        let three = self.literal_u32(3);
        let bits = self.shl_u32(bytes, three);
        self.shr_u32(value, bits)
    }

    /// Store `value` into `slot[index]`, preserving the bytes `keep` selects.
    fn merge_under_mask(
        &mut self,
        slot: u32,
        index: naga::Handle<Expression>,
        value: naga::Handle<Expression>,
        keep: naga::Handle<Expression>,
    ) -> Result<naga::Handle<Expression>, EmitError> {
        let pointer = self.binding_element_pointer_by_slot(slot, index)?;
        let existing = self.append_expr(Expression::Load { pointer });
        let overwrite = self.not_u32(keep);
        let fresh = self.and_u32(value, overwrite);
        let preserved = self.and_u32(existing, keep);
        Ok(self.or_u32(fresh, preserved))
    }
}
