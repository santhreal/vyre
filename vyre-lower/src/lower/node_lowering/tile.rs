//! Tile and matrix operation lowering into substrate-neutral kernel descriptors.

use crate::descriptor::{KernelBody, KernelOp, KernelOpKind, LiteralValue};
use crate::error::LowerError;
use vyre_foundation::ir::{BinOp, DataType, Expr, Ident, Layout, Node, SubgroupReduceOp, Tile};

use super::LowerCtx;

/// Invocations one matrix fragment is distributed across.
///
/// A tile-matmul node states a distribution so the operand words are derived
/// rather than assumed. A target whose subgroups are narrower or wider rejects
/// the op instead of reinterpreting the fragment.
const FRAGMENT_LANES: u16 = 32;

impl LowerCtx {
    pub(super) fn lower_tile_matmul(
        &mut self,
        body: &mut KernelBody,
        acc: &Ident,
        a: &Ident,
        b: &Ident,
    ) -> Result<(), LowerError> {
        let a_binding = self.scope.get_tile(a);
        let b_binding = self.scope.get_tile(b);
        let acc_binding = self.scope.get_tile(acc);

        let is_supported_mma = match (&a_binding, &b_binding, &acc_binding) {
            (Some(a_b), Some(b_b), Some(acc_b)) => {
                a_b.element == DataType::F16
                    && b_b.element == DataType::F16
                    && acc_b.element == DataType::F32
                    && a_b.extents == [16, 16]
                    && b_b.extents == [16, 8]
                    && acc_b.extents == [16, 8]
            }
            _ => false,
        };

        if is_supported_mma {
            let spec = crate::MatrixMmaSpec {
                tile: crate::MatrixTileShape { m: 16, n: 8, k: 16 },
                left: crate::FragmentValue::in_registers(
                    crate::MatrixMmaElement::F16,
                    crate::MatrixMmaLayout::RowMajor,
                    FRAGMENT_LANES,
                ),
                right: crate::FragmentValue::in_registers(
                    crate::MatrixMmaElement::F16,
                    crate::MatrixMmaLayout::ColMajor,
                    FRAGMENT_LANES,
                ),
                accumulator: crate::FragmentValue::in_registers(
                    crate::MatrixMmaElement::F32,
                    crate::MatrixMmaLayout::RowMajor,
                    FRAGMENT_LANES,
                ),
            };
            let _words = spec.operand_words().map_err(|reason| {
                LowerError::UnsupportedConstruct(format!(
                    "tile matmul declares fragments that cannot be carried: {reason}. Fix: state a tile that distributes across its lanes in whole 32-bit words."
                ))
            })?;
            let a_words = a_binding.map(|b| b.results).unwrap_or_default();
            let b_words = b_binding.map(|b| b.results).unwrap_or_default();
            let mut acc_in_words = acc_binding
                .as_ref()
                .map(|b| b.results.clone())
                .unwrap_or_default();
            let get_a =
                |this: &mut Self, body: &mut KernelBody, idx: usize| -> Result<u32, LowerError> {
                    if let Some(&id) = a_words.get(idx) {
                        Ok(id)
                    } else if let Some(scalar_id) = this.scope.get(a) {
                        Ok(scalar_id)
                    } else {
                        this.literal(body, crate::descriptor::LiteralValue::U32(0))
                    }
                };

            let get_b =
                |this: &mut Self, body: &mut KernelBody, idx: usize| -> Result<u32, LowerError> {
                    if let Some(&id) = b_words.get(idx) {
                        Ok(id)
                    } else if let Some(scalar_id) = this.scope.get(b) {
                        Ok(scalar_id)
                    } else {
                        this.literal(body, crate::descriptor::LiteralValue::U32(0))
                    }
                };

            let get_acc =
                |this: &mut Self, body: &mut KernelBody, idx: usize| -> Result<u32, LowerError> {
                    if let Some(&id) = acc_in_words.get(idx) {
                        Ok(id)
                    } else if let Some(scalar_id) = this.scope.get(acc) {
                        Ok(scalar_id)
                    } else {
                        this.literal(body, crate::descriptor::LiteralValue::F32(0.0))
                    }
                };

            let a0_low = get_a(self, body, 0)?;
            let a0_high = get_a(self, body, 1)?;
            let a0 = self.pack_f16_pair(body, a0_low, a0_high)?;

            let a1_low = get_a(self, body, 8)?;
            let a1_high = get_a(self, body, 9)?;
            let a1 = self.pack_f16_pair(body, a1_low, a1_high)?;

            let a2_low = get_a(self, body, 128)?;
            let a2_high = get_a(self, body, 129)?;
            let a2 = self.pack_f16_pair(body, a2_low, a2_high)?;

            let a3_low = get_a(self, body, 136)?;
            let a3_high = get_a(self, body, 137)?;
            let a3 = self.pack_f16_pair(body, a3_low, a3_high)?;

            let b0_low = get_b(self, body, 0)?;
            let b0_high = get_b(self, body, 1)?;
            let b0 = self.pack_f16_pair(body, b0_low, b0_high)?;

            let b1_low = get_b(self, body, 8)?;
            let b1_high = get_b(self, body, 9)?;
            let b1 = self.pack_f16_pair(body, b1_low, b1_high)?;

            let c0 = get_acc(self, body, 0)?;
            let c1 = get_acc(self, body, 1)?;
            let c2 = get_acc(self, body, 64)?;
            let c3 = get_acc(self, body, 65)?;

            let operands = vec![a0, a1, a2, a3, b0, b1, c0, c1, c2, c3];

            let result_count = spec.result_count().map_err(|reason| {
                LowerError::UnsupportedConstruct(format!(
                    "tile matmul declares invalid result fragment: {reason}"
                ))
            })?;
            let base_result_id = self.alloc_values(result_count)?;
            let result_ids: Vec<u32> = (0..result_count).map(|i| base_result_id + i).collect();

            body.ops.push(KernelOp {
                kind: KernelOpKind::MatrixMma(Box::new(spec)),
                operands,
                result: Some(base_result_id),
            });

            while acc_in_words.len() < 128 {
                let zero_id = self.literal(body, LiteralValue::F32(0.0))?;
                acc_in_words.push(zero_id);
            }
            acc_in_words[0] = result_ids[0];
            acc_in_words[1] = result_ids[1];
            acc_in_words[64] = result_ids[2];
            acc_in_words[65] = result_ids[3];

            self.scope
                .bind_tile(acc.clone(), vec![16, 8], DataType::F32, acc_in_words);
            Ok(())
        } else {
            let a_elems = a_binding.map(|b| b.results).unwrap_or_default();
            let b_elems = b_binding.map(|b| b.results).unwrap_or_default();
            let mut acc_elems = acc_binding
                .as_ref()
                .map(|b| b.results.clone())
                .unwrap_or_default();
            let elem_type = acc_binding
                .as_ref()
                .map(|b| b.element.clone())
                .unwrap_or(DataType::F32);

            let a_len = a_elems.len();
            let b_len = b_elems.len();
            let (m, k, n) = if a_len == 16 * 16 && b_len == 16 * 8 {
                (16, 16, 8)
            } else if a_len == 16 * 8 && b_len == 8 * 16 {
                (16, 8, 16)
            } else {
                let k = (a_len as f64).sqrt().round() as usize;
                let k = if k == 0 { 1 } else { k };
                let m = a_len / k;
                let n = if k > 0 { b_len / k } else { 1 };
                (m.max(1), k.max(1), n.max(1))
            };

            while acc_elems.len() < m * n {
                let zero_lit = match elem_type {
                    DataType::F32 => LiteralValue::F32(0.0),
                    DataType::I32 => LiteralValue::I32(0),
                    DataType::Bool => LiteralValue::Bool(false),
                    _ => LiteralValue::U32(0),
                };
                let zero_id = self.literal(body, zero_lit)?;
                acc_elems.push(zero_id);
            }

            for i in 0..m {
                for j in 0..n {
                    let acc_idx = i * n + j;
                    let mut current_sum_id = acc_elems[acc_idx];
                    for p in 0..k {
                        let a_idx = i * k + p;
                        let b_idx = p * n + j;
                        let a_id = *a_elems.get(a_idx).ok_or_else(|| {
                            LowerError::UnsupportedConstruct(format!(
                                "tile matmul reads element {a_idx} of a left operand holding {a_len}. Fix: bind the left tile at the extents the product derives."
                            ))
                        })?;
                        let b_id = *b_elems.get(b_idx).ok_or_else(|| {
                            LowerError::UnsupportedConstruct(format!(
                                "tile matmul reads element {b_idx} of a right operand holding {b_len}. Fix: bind the right tile at the extents the product derives."
                            ))
                        })?;
                        let prod_id =
                            self.binary(body, KernelOpKind::BinOpKind(BinOp::Mul), a_id, b_id)?;
                        current_sum_id = self.binary(
                            body,
                            KernelOpKind::BinOpKind(BinOp::Add),
                            current_sum_id,
                            prod_id,
                        )?;
                    }
                    acc_elems[acc_idx] = current_sum_id;
                }
            }
            self.scope
                .bind_tile(acc.clone(), vec![m as u32, n as u32], elem_type, acc_elems);
            Ok(())
        }
    }

    pub(super) fn lower_tile_load(
        &mut self,
        body: &mut KernelBody,
        tile: &Ident,
        tile_type: &Tile,
        buffer: &Ident,
        origin: &[Expr],
        layout: &Layout,
    ) -> Result<(), LowerError> {
        let slot = self.buffer_slot(buffer)?;
        let mut origin_ids = Vec::with_capacity(origin.len());
        for expr in origin {
            origin_ids.push(self.lower_expr(expr, body)?);
        }

        let total_elements = tile_type.element_count();
        let mut elements: Vec<Option<u32>> = vec![None; total_elements];

        let mut strides = vec![1u32; tile_type.extents.len()];
        for i in (0..tile_type.extents.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * tile_type.extents[i + 1];
        }

        if tile_type.extents.is_empty() {
            let global_idx_id = match origin_ids.first() {
                Some(&id) => id,
                None => self.literal(body, LiteralValue::U32(0))?,
            };
            let result_id = self.alloc_value()?;
            body.ops.push(KernelOp {
                kind: self.load_kind(slot),
                operands: vec![slot, global_idx_id],
                result: Some(result_id),
            });
            elements = vec![Some(result_id)];
        } else if tile_type.extents.len() == 1 {
            let n = tile_type.extents[0];
            let base_id = if let Some(&first) = origin_ids.first() {
                first
            } else {
                self.literal(body, LiteralValue::U32(0))?
            };
            for i in 0..n {
                let global_idx_id = if i == 0 {
                    base_id
                } else {
                    let i_lit = self.literal(body, LiteralValue::U32(i))?;
                    self.binary(body, KernelOpKind::BinOpKind(BinOp::Add), base_id, i_lit)?
                };
                let result_id = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: self.load_kind(slot),
                    operands: vec![slot, global_idx_id],
                    result: Some(result_id),
                });
                let local_idx = layout.linear_index(&[i], &tile_type.extents);
                if local_idx < elements.len() {
                    elements[local_idx] = Some(result_id);
                }
            }
        } else if tile_type.extents.len() == 2 {
            let rows = tile_type.extents[0];
            let cols = tile_type.extents[1];
            let r_base_id = if let Some(&r) = origin_ids.first() {
                r
            } else {
                self.literal(body, LiteralValue::U32(0))?
            };
            let c_base_id = if let Some(&c) = origin_ids.get(1) {
                c
            } else {
                self.literal(body, LiteralValue::U32(0))?
            };
            let cols_lit = self.literal(body, LiteralValue::U32(cols))?;
            for r in 0..rows {
                let r_offset = if r == 0 {
                    r_base_id
                } else {
                    let r_lit = self.literal(body, LiteralValue::U32(r))?;
                    self.binary(body, KernelOpKind::BinOpKind(BinOp::Add), r_base_id, r_lit)?
                };
                let row_times_cols = self.binary(
                    body,
                    KernelOpKind::BinOpKind(BinOp::Mul),
                    r_offset,
                    cols_lit,
                )?;
                for c in 0..cols {
                    let c_offset = if c == 0 {
                        c_base_id
                    } else {
                        let c_lit = self.literal(body, LiteralValue::U32(c))?;
                        self.binary(body, KernelOpKind::BinOpKind(BinOp::Add), c_base_id, c_lit)?
                    };
                    let global_idx_id = self.binary(
                        body,
                        KernelOpKind::BinOpKind(BinOp::Add),
                        row_times_cols,
                        c_offset,
                    )?;
                    let result_id = self.alloc_value()?;
                    body.ops.push(KernelOp {
                        kind: self.load_kind(slot),
                        operands: vec![slot, global_idx_id],
                        result: Some(result_id),
                    });
                    let local_idx = layout.linear_index(&[r, c], &tile_type.extents);
                    if local_idx < elements.len() {
                        elements[local_idx] = Some(result_id);
                    }
                }
            }
        } else {
            for idx in 0..total_elements {
                let mut coords = Vec::with_capacity(tile_type.extents.len());
                let mut temp = idx as u32;
                for &extent in tile_type.extents.iter().rev() {
                    coords.push(temp % extent);
                    temp /= extent;
                }
                coords.reverse();

                let mut sum_id: Option<u32> = None;
                for (i, &coord_c) in coords.iter().enumerate() {
                    let base_id = if let Some(&b) = origin_ids.get(i) {
                        b
                    } else {
                        self.literal(body, LiteralValue::U32(0))?
                    };
                    let c_offset = if coord_c == 0 {
                        base_id
                    } else {
                        let c_lit = self.literal(body, LiteralValue::U32(coord_c))?;
                        self.binary(body, KernelOpKind::BinOpKind(BinOp::Add), base_id, c_lit)?
                    };
                    let term = if strides[i] == 1 {
                        c_offset
                    } else {
                        let stride_lit = self.literal(body, LiteralValue::U32(strides[i]))?;
                        self.binary(
                            body,
                            KernelOpKind::BinOpKind(BinOp::Mul),
                            c_offset,
                            stride_lit,
                        )?
                    };
                    sum_id = match sum_id {
                        None => Some(term),
                        Some(prev) => Some(self.binary(
                            body,
                            KernelOpKind::BinOpKind(BinOp::Add),
                            prev,
                            term,
                        )?),
                    };
                }
                let global_idx_id = if let Some(sum) = sum_id {
                    sum
                } else {
                    self.literal(body, LiteralValue::U32(0))?
                };
                let result_id = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: self.load_kind(slot),
                    operands: vec![slot, global_idx_id],
                    result: Some(result_id),
                });
                let local_idx = layout.linear_index(&coords, &tile_type.extents);
                if local_idx < elements.len() {
                    elements[local_idx] = Some(result_id);
                }
            }
        }
        let elements = elements
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                id.ok_or_else(|| {
                    LowerError::UnsupportedConstruct(format!(
                        "tile load leaves element {index} of `{tile}` unwritten. Fix: state a layout whose linear index covers every element of the tile extents."
                    ))
                })
            })
            .collect::<Result<Vec<u32>, LowerError>>()?;
        self.scope.bind_tile(
            tile.clone(),
            tile_type.extents.clone(),
            tile_type.element.clone(),
            elements,
        );
        Ok(())
    }

    pub(super) fn lower_tile_store(
        &mut self,
        body: &mut KernelBody,
        buffer: &Ident,
        origin: &[Expr],
        tile: &Ident,
    ) -> Result<(), LowerError> {
        let slot = self.buffer_slot(buffer)?;
        let store_kind = self.store_kind(slot, buffer)?;
        let origin_id = if let Some(first) = origin.first() {
            self.lower_expr(first, body)?
        } else {
            self.literal(body, LiteralValue::U32(0))?
        };
        let tile_words = self
            .scope
            .get_tile(tile)
            .map(|b| b.results)
            .unwrap_or_else(|| self.scope.get(tile).into_iter().collect());
        for (i, word_id) in tile_words.iter().enumerate() {
            let global_idx_id = if i == 0 {
                origin_id
            } else {
                let i_lit = self.literal(body, LiteralValue::U32(i as u32))?;
                self.binary(body, KernelOpKind::BinOpKind(BinOp::Add), origin_id, i_lit)?
            };
            body.ops.push(KernelOp {
                kind: store_kind.clone(),
                operands: vec![slot, global_idx_id, *word_id],
                result: None,
            });
        }
        Ok(())
    }

    pub(super) fn lower_tile_reduce(
        &mut self,
        body: &mut KernelBody,
        out: &Ident,
        tile: &Ident,
        op: SubgroupReduceOp,
        axis: u32,
    ) -> Result<(), LowerError> {
        let tile_bind = self.scope.get_tile(tile);
        let elements = tile_bind
            .as_ref()
            .map(|b| b.results.clone())
            .unwrap_or_else(|| self.scope.get(tile).into_iter().collect());
        let elem_type = tile_bind
            .as_ref()
            .map(|b| b.element.clone())
            .unwrap_or(DataType::F32);

        let bin_op = match op {
            SubgroupReduceOp::Add => BinOp::Add,
            SubgroupReduceOp::Mul => BinOp::Mul,
            SubgroupReduceOp::Min => BinOp::Min,
            SubgroupReduceOp::Max => BinOp::Max,
            SubgroupReduceOp::And => BinOp::BitAnd,
            SubgroupReduceOp::Or => BinOp::BitOr,
            SubgroupReduceOp::Xor => BinOp::BitXor,
            _ => BinOp::Add,
        };

        let total = elements.len();
        let dim = (total as f64).sqrt().round() as usize;
        let (rows, cols) = if dim * dim == total && dim > 0 {
            (dim, dim)
        } else if total % 2 == 0 {
            (total / 2, 2)
        } else {
            (total, 1)
        };

        let mut res = Vec::new();
        let out_extents;

        if axis == 1 && rows > 0 && cols > 0 && rows * cols == total {
            out_extents = vec![rows as u32];
            for r in 0..rows {
                let slice = &elements[r * cols..(r + 1) * cols];
                let reduced = self.reduce_slice_ops(body, slice, bin_op, &elem_type)?;
                res.push(reduced);
            }
        } else if axis == 0 && rows > 0 && cols > 0 && rows * cols == total {
            out_extents = vec![cols as u32];
            for c in 0..cols {
                let col_vals: Vec<u32> = (0..rows).map(|r| elements[r * cols + c]).collect();
                let reduced = self.reduce_slice_ops(body, &col_vals, bin_op, &elem_type)?;
                res.push(reduced);
            }
        } else {
            out_extents = vec![1];
            let reduced = self.reduce_slice_ops(body, &elements, bin_op, &elem_type)?;
            res.push(reduced);
        }

        self.scope
            .bind_tile(out.clone(), out_extents, elem_type, res);
        Ok(())
    }

    pub(super) fn lower_tile_elementwise(
        &mut self,
        body: &mut KernelBody,
        depth: usize,
        out: &Ident,
        inputs: &[Ident],
        inner_body: &[Node],
    ) -> Result<(), LowerError> {
        let mut input_bindings = Vec::with_capacity(inputs.len());
        let mut max_len = 0;
        let mut result_extents = Vec::new();
        let mut result_elem_type = DataType::F32;

        for input in inputs {
            let binding = self.scope.get_tile(input).ok_or_else(|| {
                LowerError::InvalidProgram(format!(
                    "tile input `{input}` is referenced before binding. Fix: emit a tile declaration or load before use."
                ))
            })?;
            let len = binding.results.len();
            if len > max_len {
                max_len = len;
                result_extents = binding.extents.clone();
                result_elem_type = binding.element.clone();
            }
            input_bindings.push(binding);
        }

        for (input, binding) in inputs.iter().zip(&input_bindings) {
            let n = binding.results.len();
            if n == 0 || (max_len > 0 && max_len % n != 0) {
                return Err(LowerError::UnsupportedConstruct(format!(
                    "tile elementwise input `{input}` length {n} does not divide output length {max_len}. Fix: ensure tile elementwise input dimensions broadcast evenly into the output shape."
                )));
            }
        }

        let saved_bindings: Vec<(Ident, Option<super::super::scope::TileBinding>, Option<u32>)> =
            inputs
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        self.scope.get_tile(name),
                        self.scope.get(name),
                    )
                })
                .collect();

        let mut out_elems = Vec::with_capacity(max_len);
        for idx in 0..max_len {
            for (i, input) in inputs.iter().enumerate() {
                let n = input_bindings[i].results.len();
                let elem_idx = if n > 0 { idx / (max_len / n) } else { 0 };
                let elem_id = input_bindings[i].results[elem_idx];
                self.scope.bind(input.clone(), elem_id);
            }
            self.lower_nodes(inner_body, body, depth)?;
            let out_val_id = self.scope.get(out).ok_or_else(|| {
                LowerError::InvalidProgram(format!(
                    "tile elementwise body did not bind output variable `{out}`. Fix: bind `{out}` in the elementwise body."
                ))
            })?;
            out_elems.push(out_val_id);
        }

        for (name, tile_bind, scalar_bind) in saved_bindings {
            if let Some(tb) = tile_bind {
                self.scope
                    .bind_tile(name, tb.extents, tb.element, tb.results);
            } else if let Some(sb) = scalar_bind {
                self.scope.bind(name, sb);
            }
        }

        self.scope
            .bind_tile(out.clone(), result_extents, result_elem_type, out_elems);
        Ok(())
    }

    pub(super) fn lower_tile_decl(
        &mut self,
        body: &mut KernelBody,
        name: &Ident,
        tile: &Tile,
    ) -> Result<(), LowerError> {
        let total_elements = tile.element_count();
        let zero_lit = match tile.element {
            DataType::F32 => LiteralValue::F32(0.0),
            DataType::I32 => LiteralValue::I32(0),
            DataType::Bool => LiteralValue::Bool(false),
            _ => LiteralValue::U32(0),
        };
        let mut word_ids = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let res_id = self.literal(body, zero_lit.clone())?;
            word_ids.push(res_id);
        }
        self.scope.bind_tile(
            name.clone(),
            tile.extents.clone(),
            tile.element.clone(),
            word_ids,
        );
        Ok(())
    }

    fn reduce_slice_ops(
        &mut self,
        body: &mut KernelBody,
        slice: &[u32],
        bin_op: BinOp,
        elem_type: &DataType,
    ) -> Result<u32, LowerError> {
        if slice.is_empty() {
            let zero_lit = match elem_type {
                DataType::F32 => LiteralValue::F32(0.0),
                DataType::I32 => LiteralValue::I32(0),
                DataType::Bool => LiteralValue::Bool(false),
                _ => LiteralValue::U32(0),
            };
            return self.literal(body, zero_lit);
        }
        let mut acc = slice[0];
        for &elem in &slice[1..] {
            acc = self.binary(body, KernelOpKind::BinOpKind(bin_op), acc, elem)?;
        }
        Ok(acc)
    }

    fn pack_f16_pair(
        &mut self,
        body: &mut KernelBody,
        low_id: u32,
        high_id: u32,
    ) -> Result<u32, LowerError> {
        let shift_16 = self.literal(body, LiteralValue::U32(16))?;
        let mask_16 = self.literal(body, LiteralValue::U32(0xFFFF))?;
        let low_u32 = self.unary(
            body,
            KernelOpKind::Cast {
                target: DataType::U32,
            },
            low_id,
        )?;
        let low_masked = self.binary(
            body,
            KernelOpKind::BinOpKind(BinOp::BitAnd),
            low_u32,
            mask_16,
        )?;
        let high_u32 = self.unary(
            body,
            KernelOpKind::Cast {
                target: DataType::U32,
            },
            high_id,
        )?;
        let high_shifted = self.binary(
            body,
            KernelOpKind::BinOpKind(BinOp::Shl),
            high_u32,
            shift_16,
        )?;
        self.binary(
            body,
            KernelOpKind::BinOpKind(BinOp::BitOr),
            low_masked,
            high_shifted,
        )
    }
}
