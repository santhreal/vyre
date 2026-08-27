//! Type, signature, and body feature classification for host oracle detection.

use std::collections::BTreeSet;
use syn::visit::Visit;

use super::host_oracle_elimination_extract::{extract_pat_bindings, IdentReadCollector};
use super::host_oracle_elimination_records::{
    compares_operands, computes_from_operands, SCALAR_TYPES,
};

/// Recursively check if type is a scalar or data collection.
pub(super) fn type_is_data_output(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(seg) = type_path.path.segments.last() {
                let ident = seg.ident.to_string();
                if SCALAR_TYPES.contains(&ident.as_str()) {
                    return true;
                }
                if ident == "Vec"
                    || ident == "BTreeSet"
                    || ident == "HashSet"
                    || ident == "BTreeMap"
                    || ident == "HashMap"
                {
                    return true;
                }
                if ident == "Result"
                    || ident == "Option"
                    || ident == "Arc"
                    || ident == "Box"
                    || ident == "Rc"
                {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            return type_is_data_output(inner_ty);
                        }
                    }
                    return false;
                }
            }
            false
        }
        syn::Type::ImplTrait(_) => true,
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_data_output),
        syn::Type::Array(a) => type_is_data_output(&a.elem),
        syn::Type::Slice(s) => type_is_data_output(&s.elem),
        syn::Type::Reference(r) => type_is_data_output(&r.elem),
        _ => false,
    }
}

/// Check if return type is a heap/collection data container.
pub(super) fn has_data_container_output(sig: &syn::Signature) -> bool {
    match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => type_is_data_container(ty),
    }
}

pub(super) fn type_is_data_container(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(seg) = type_path.path.segments.last() {
                let ident = seg.ident.to_string();
                if ident == "Vec"
                    || ident == "BTreeSet"
                    || ident == "HashSet"
                    || ident == "BTreeMap"
                    || ident == "HashMap"
                {
                    return true;
                }
                if ident == "Result"
                    || ident == "Option"
                    || ident == "Arc"
                    || ident == "Box"
                    || ident == "Rc"
                {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            return type_is_data_container(inner_ty);
                        }
                    }
                }
            }
            false
        }
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_data_container),
        syn::Type::Array(_) | syn::Type::Slice(_) => true,
        syn::Type::Reference(r) => type_is_data_container(&r.elem),
        _ => false,
    }
}

/// Check if the signature accepts primitive numeric payload data.
pub(super) fn has_data_input_ast(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Typed(pat_type) => type_is_numeric_payload_input(&pat_type.ty),
        syn::FnArg::Receiver(_) => false,
    })
}

pub(super) fn type_is_numeric_payload_input(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(reference) => match &*reference.elem {
            syn::Type::Slice(s) => type_is_scalar_or_payload(&s.elem),
            syn::Type::Array(a) => type_is_scalar_or_payload(&a.elem),
            syn::Type::Path(path) => {
                let Some(segment) = path.path.segments.last() else {
                    return false;
                };
                let ident = segment.ident.to_string();
                if matches!(ident.as_str(), "Vec" | "BTreeSet" | "HashSet") {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        return args.args.iter().any(|arg| {
                            matches!(
                                arg,
                                syn::GenericArgument::Type(inner)
                                    if type_is_scalar_or_payload(inner)
                            )
                        });
                    }
                    return true;
                }
                false
            }
            _ => type_is_numeric_payload_input(&reference.elem),
        },
        syn::Type::Slice(slice) => type_is_scalar_or_payload(&slice.elem),
        syn::Type::Array(array) => type_is_scalar_or_payload(&array.elem),
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_numeric_payload_input),
        syn::Type::Group(group) => type_is_numeric_payload_input(&group.elem),
        syn::Type::Paren(paren) => type_is_numeric_payload_input(&paren.elem),
        syn::Type::Path(path) => {
            if type_is_scalar_or_payload(ty) {
                return true;
            }
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            if matches!(ident.as_str(), "Vec" | "BTreeSet" | "HashSet") {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    return args.args.iter().any(|arg| {
                        matches!(
                            arg,
                            syn::GenericArgument::Type(inner)
                                if type_is_scalar_or_payload(inner)
                        )
                    });
                }
                return true;
            }
            if matches!(ident.as_str(), "Option" | "Result" | "Arc" | "Box" | "Rc") {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    return args.args.iter().any(|arg| {
                        matches!(
                            arg,
                            syn::GenericArgument::Type(inner)
                                if type_is_numeric_payload_input(inner)
                        )
                    });
                }
            }
            false
        }
        _ => false,
    }
}

pub(super) fn type_is_scalar_or_payload(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) => {
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            SCALAR_TYPES.contains(&ident.as_str())
                || matches!(ident.as_str(), "Vec" | "u8" | "u32" | "f32")
        }
        syn::Type::Array(_) | syn::Type::Slice(_) => true,
        syn::Type::Reference(r) => type_is_scalar_or_payload(&r.elem),
        syn::Type::Tuple(t) => t.elems.iter().any(type_is_scalar_or_payload),
        _ => false,
    }
}

pub(super) fn type_is_data_output_ret(ret: &syn::ReturnType) -> bool {
    match ret {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => type_is_data_output(ty),
    }
}

/// Check if signature returns a non-unit value.
pub(super) fn has_data_output_ast(sig: &syn::Signature) -> bool {
    match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Tuple(t) if t.elems.is_empty() => false,
            _ => true,
        },
    }
}

/// Check if return type is unit `()`.
pub(super) fn returns_unit(ret: &syn::ReturnType) -> bool {
    match ret {
        syn::ReturnType::Default => true,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Tuple(t) => t.elems.is_empty(),
            _ => false,
        },
    }
}

/// Check if return type is `Result<(), E>`.
pub(super) fn returns_result_unit(ret: &syn::ReturnType) -> bool {
    match ret {
        syn::ReturnType::Type(_, ty) => is_result_unit(ty),
        _ => false,
    }
}

pub(super) fn is_result_unit(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            if seg.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(ok_ty)) = args.args.first() {
                        return match ok_ty {
                            syn::Type::Tuple(t) => t.elems.is_empty(),
                            _ => false,
                        };
                    }
                }
            }
        }
    }
    false
}

pub(super) fn is_dispatch_sizing_or_validator(sig: &syn::Signature) -> bool {
    let fn_name = sig.ident.to_string();
    if fn_name.starts_with("ceil_div")
        || fn_name.starts_with("div_ceil")
        || fn_name.ends_with("_key")
        || fn_name.ends_with("_hash")
        || fn_name.ends_with("_fingerprint")
        || fn_name.ends_with("_bytes")
        || fn_name.ends_with("_size")
        || fn_name.ends_with("_len")
        || fn_name.ends_with("_count")
        || fn_name.starts_with("count_")
        || fn_name.ends_with("_capacity")
        || fn_name.ends_with("_scratch")
        || fn_name.ends_with("_scratch_bytes")
        || fn_name.ends_with("_offset")
        || fn_name.ends_with("_offsets")
        || fn_name.ends_with("_stride")
        || fn_name.ends_with("_strides")
        || fn_name.ends_with("_alignment")
        || fn_name.ends_with("_align")
        || fn_name.ends_with("_shape")
        || fn_name.ends_with("_layout")
        || fn_name.ends_with("_stats")
        || fn_name.ends_with("_evidence")
        || fn_name.ends_with("_planner_evidence")
        || fn_name == "static_input_key"
        || fn_name == "program_cache_key"
        || fn_name == "persistent_bfs_single_program_cache_key"
        || fn_name == "persistent_bfs_batch_program_cache_key"
        || fn_name == "persistent_bfs_program_layout_hash"
        || fn_name == "dominator_frontier_slice_fingerprint"
        || fn_name == "u32_slice_fingerprint"
        || fn_name == "padded_u32_slice_fingerprint"
        || fn_name == "path_reconstruct_u32_slice_fingerprint"
        || fn_name == "toposort_csr_slice_fingerprint"
        || fn_name == "fingerprint_u32_slice"
        || fn_name == "slots"
        || fn_name == "density_bps"
        || fn_name == "cell_count"
        || fn_name == "mp_upper_edge"
        || fn_name == "sum_product_depths"
        || fn_name == "auto_tile_for"
        || fn_name == "action_at"
        || fn_name == "goto_at"
        || fn_name == "counts"
        || fn_name == "get_or_insert_with"
        || fn_name == "try_get_or_insert_with"
        || fn_name == "parse_corpus_parallel"
        || fn_name == "ensure"
        || fn_name == "ensure_resident_query_handles"
        || fn_name == "extend_handles"
        || fn_name == "free"
        || fn_name == "from_rules"
        || fn_name == "run_csr_bidirectional_closure_plan_with_step"
        || fn_name == "merge_frontier_or_changed"
        || fn_name == "try_merge_frontier_or_changed"
        || fn_name == "plan_budgeted_effective_chunks"
        || fn_name == "resident_csr_queue_frontier_stats"
        || fn_name == "insert"
        || fn_name == "contains"
        || fn_name == "intersects"
        || fn_name == "write_pivot_bitsets"
        || fn_name == "copy_csr_forward_seed_frontier_into"
        || fn_name == "plan_persistent_bfs_dispatch"
        || fn_name == "pack_dispatch_table"
        || fn_name == "pack_dispatch_table_into"
        || fn_name == "unpack_entry"
        || fn_name == "prepare"
        || fn_name == "prepare_ifds_rule_columns"
        || fn_name == "split_ifds_rules_into"
        || fn_name == "split_ifds_rule_triples_into"
        || fn_name == "split_ifds_rule_quads_into"
        || fn_name == "canonicalize_csr_within_rows_in_place"
        || fn_name == "admissible_workgroup_widths"
        || fn_name == "dispatch_resident_timed"
        || fn_name == "should_split_grid_sync"
        || fn_name == "contains_grid_sync"
        || fn_name == "fnv1a64_mix_u32"
        || fn_name == "fnv1a64_initial_state"
        || fn_name == "split_regions_into"
        || fn_name == "split_regions"
        || fn_name == "intersects"
    {
        return true;
    }

    let has_collection_inputs =
        sig.inputs.iter().any(|arg| match arg {
            syn::FnArg::Typed(pat) => match &*pat.ty {
                syn::Type::Slice(_) => true,
                syn::Type::Reference(r) => matches!(&*r.elem, syn::Type::Slice(_)),
                syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                    s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                }),
                _ => false,
            },
            _ => false,
        });
    if has_collection_inputs {
        return false;
    }

    match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Path(p) => {
                let Some(seg) = p.path.segments.last() else {
                    return false;
                };
                let ident = seg.ident.to_string();
                if ident == "Result" {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(ok_ty)) = args.args.first() {
                            let ok_ident = match ok_ty {
                                syn::Type::Path(p) => {
                                    p.path.segments.last().map(|s| s.ident.to_string())
                                }
                                syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
                                syn::Type::Array(_) => Some("array".to_string()),
                                _ => None,
                            };
                            if let Some(id) = ok_ident {
                                return matches!(
                                    id.as_str(),
                                    "()" | "bool"
                                        | "usize"
                                        | "u32"
                                        | "u64"
                                        | "u16"
                                        | "u8"
                                        | "i32"
                                        | "i64"
                                        | "array"
                                        | "Option"
                                );
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        },
    }
}

/// Check if signature has mutable parameters.
pub(super) fn has_mutable_params(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Receiver(r) => r.mutability.is_some(),
        syn::FnArg::Typed(pat_type) => match &*pat_type.ty {
            syn::Type::Reference(r) => r.mutability.is_some(),
            _ => false,
        },
    })
}

/// Check if signature has mutable data output parameters.
pub(super) fn has_mutable_data_output_param(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Receiver(r) => r.mutability.is_some(),
        syn::FnArg::Typed(pat_type) => match &*pat_type.ty {
            syn::Type::Reference(r) => {
                r.mutability.is_some()
                    && !type_is_byte_buffer(&r.elem)
                    && type_is_data_output(&r.elem)
            }
            _ => false,
        },
    })
}

/// Check if signature is a Formatter formatting method `fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result`.
pub(super) fn is_fmt_signature(sig: &syn::Signature) -> bool {
    let is_fmt_ret = match &sig.output {
        syn::ReturnType::Type(_, ty) => {
            if let syn::Type::Path(p) = &**ty {
                p.path.segments.last().is_some_and(|s| s.ident == "Result")
            } else {
                false
            }
        }
        _ => false,
    };
    let has_formatter_param = sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            let ty_str = quote::quote!(#pat_type.ty).to_string().replace(' ', "");
            ty_str.contains("Formatter")
        } else {
            false
        }
    });
    is_fmt_ret && has_formatter_param
}

pub(super) fn is_byte_unpack_codec_expr(expr: &syn::ExprBinary) -> bool {
    match expr.op {
        syn::BinOp::Shl(_)
        | syn::BinOp::ShlAssign(_)
        | syn::BinOp::Shr(_)
        | syn::BinOp::ShrAssign(_) => {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(lit_int),
                ..
            }) = &*expr.right
            {
                let val = lit_int.base10_parse::<u32>().unwrap_or(0);
                val == 8
                    || val == 16
                    || val == 24
                    || val == 32
                    || val == 40
                    || val == 48
                    || val == 56
            } else {
                false
            }
        }
        syn::BinOp::BitOr(_)
        | syn::BinOp::BitOrAssign(_)
        | syn::BinOp::BitAnd(_)
        | syn::BinOp::BitAndAssign(_)
        | syn::BinOp::BitXor(_)
        | syn::BinOp::BitXorAssign(_) => true,
        _ => false,
    }
}

pub(super) fn has_byte_buffer_param(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            type_is_byte_buffer(&pat_type.ty)
        } else {
            false
        }
    })
}

pub(super) fn type_is_byte_buffer(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(r) => type_is_byte_buffer(&r.elem),
        syn::Type::Slice(s) => type_is_byte_buffer(&s.elem),
        syn::Type::Path(p) => {
            if p.path.is_ident("u8") {
                return true;
            }
            if let Some(seg) = p.path.segments.last() {
                if seg.ident == "Vec" || seg.ident == "SmallVec" {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            return type_is_byte_buffer(inner_ty);
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

pub(super) fn type_is_direct_numeric_scalar(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(reference) => type_is_direct_numeric_scalar(&reference.elem),
        syn::Type::Group(group) => type_is_direct_numeric_scalar(&group.elem),
        syn::Type::Paren(paren) => type_is_direct_numeric_scalar(&paren.elem),
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| SCALAR_TYPES.contains(&segment.ident.to_string().as_str())),
        _ => false,
    }
}

pub(super) struct WireCodecSemanticVisitor {
    pub(super) semantic_idents: BTreeSet<String>,
    pub(super) forbidden: bool,
}

impl<'ast> Visit<'ast> for WireCodecSemanticVisitor {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        let is_arithmetic = matches!(
            expr.op,
            syn::BinOp::Add(_)
                | syn::BinOp::Sub(_)
                | syn::BinOp::Mul(_)
                | syn::BinOp::Div(_)
                | syn::BinOp::Rem(_)
                | syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
        );
        if is_arithmetic && !is_byte_unpack_codec_expr(expr) {
            let mut reads = IdentReadCollector::default();
            reads.visit_expr_binary(expr);
            if reads
                .idents
                .iter()
                .any(|ident| self.semantic_idents.contains(ident))
            {
                self.forbidden = true;
            }
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_unary(&mut self, expr: &'ast syn::ExprUnary) {
        let mut reads = IdentReadCollector::default();
        reads.visit_expr_unary(expr);
        if reads
            .idents
            .iter()
            .any(|ident| self.semantic_idents.contains(ident))
        {
            self.forbidden = true;
        }
        syn::visit::visit_expr_unary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method = expr.method.to_string();
        if matches!(
            method.as_str(),
            "sum"
                | "product"
                | "fold"
                | "rfold"
                | "reduce"
                | "wrapping_add"
                | "wrapping_sub"
                | "wrapping_mul"
                | "wrapping_div"
                | "wrapping_rem"
                | "overflowing_add"
                | "overflowing_sub"
                | "overflowing_mul"
                | "count_ones"
                | "count_zeros"
                | "rotate_left"
                | "rotate_right"
                | "reverse_bits"
                | "abs"
                | "sqrt"
                | "powf"
                | "powi"
                | "exp"
                | "ln"
                | "sin"
                | "cos"
                | "tan"
        ) {
            self.forbidden = true;
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        let is_index_range = matches!(&*expr.expr, syn::Expr::Range(_));
        if is_index_range {
            syn::visit::visit_expr_for_loop(self, expr);
            return;
        }
        let mut loop_bindings: BTreeSet<String> = BTreeSet::new();
        extract_pat_bindings(&expr.pat, &mut loop_bindings);
        let added = loop_bindings
            .iter()
            .filter(|binding| self.semantic_idents.insert((*binding).clone()))
            .cloned()
            .collect::<Vec<_>>();
        syn::visit::visit_expr(&mut *self, &expr.expr);
        self.visit_block(&expr.body);
        for binding in added {
            self.semantic_idents.remove(&binding);
        }
    }
}

pub(super) fn is_wire_codec_ast(sig: &syn::Signature, block: &syn::Block) -> bool {
    if !has_byte_buffer_param(sig) {
        return false;
    }

    let mut semantic_idents: BTreeSet<String> = BTreeSet::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if !type_is_byte_buffer(&pat_type.ty)
                && (type_is_direct_numeric_scalar(&pat_type.ty)
                    || type_is_numeric_payload_input(&pat_type.ty))
            {
                extract_pat_bindings(&pat_type.pat, &mut semantic_idents);
            }
        }
    }
    let mut visitor = WireCodecSemanticVisitor {
        semantic_idents,
        forbidden: false,
    };
    visitor.visit_block(block);
    !visitor.forbidden
}

/// AST visitor extracting structural mathematical, arithmetic, loop, and algorithmic features from function bodies.
#[derive(Default)]
pub(super) struct BodyFeatureVisitor {
    pub(super) has_payload_arithmetic: bool,
    pub(super) has_comparison: bool,
    pub(super) has_unary_op: bool,
    pub(super) has_numeric_method: bool,
    pub(super) has_branch_on_data: bool,
    pub(super) has_loops: bool,
    pub(super) has_iterators_or_transforms: bool,
    pub(super) has_algorithms: bool,
}

impl BodyFeatureVisitor {
    pub(super) fn has_semantic_operation(&self) -> bool {
        self.has_payload_arithmetic
            || self.has_comparison
            || self.has_unary_op
            || self.has_numeric_method
            || self.has_loops
            || self.has_iterators_or_transforms
            || self.has_algorithms
            || self.has_branch_on_data
    }
}

impl<'ast> Visit<'ast> for BodyFeatureVisitor {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if computes_from_operands(&expr.op) {
            self.has_payload_arithmetic = true;
        } else if compares_operands(&expr.op) {
            self.has_comparison = true;
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_unary(&mut self, expr: &'ast syn::ExprUnary) {
        match expr.op {
            syn::UnOp::Neg(_) | syn::UnOp::Not(_) => {
                self.has_unary_op = true;
            }
            _ => {}
        }
        syn::visit::visit_expr_unary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let name = expr.method.to_string();
        match name.as_str() {
            "min" | "max" | "clamp" | "abs" | "round" | "floor" | "ceil" | "trunc" | "fract"
            | "signum" | "recip" | "sqrt" | "cbrt" | "powf" | "powi" | "exp" | "exp2" | "ln"
            | "log" | "log2" | "log10" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "atan2" | "sinh" | "cosh" | "tanh" | "count_ones" | "count_zeros"
            | "leading_zeros" | "trailing_zeros" | "leading_ones" | "trailing_ones"
            | "rotate_left" | "rotate_right" | "reverse_bits" | "swap_bytes" | "wrapping_add"
            | "wrapping_sub" | "wrapping_mul" | "wrapping_div" | "wrapping_rem"
            | "wrapping_neg" | "wrapping_shl" | "wrapping_shr" | "saturating_add"
            | "saturating_sub" | "saturating_mul" | "saturating_div" | "saturating_pow"
            | "overflowing_add" | "overflowing_sub" | "overflowing_mul" | "checked_add"
            | "checked_sub" | "checked_mul" | "checked_div" | "sum" | "product" | "count"
            | "any" | "all" | "mul_add" | "copysign" => {
                self.has_numeric_method = true;
            }
            "map" | "filter" | "filter_map" | "fold" | "rfold" | "reduce" | "for_each" | "zip"
            | "enumerate" | "step_by" | "chain" | "flatten" | "flat_map" => {
                self.has_iterators_or_transforms = true;
            }
            "binary_search"
            | "binary_search_by"
            | "binary_search_by_key"
            | "partition_point"
            | "sort"
            | "sort_unstable"
            | "sort_by"
            | "sort_unstable_by"
            | "sort_by_key"
            | "sort_unstable_by_key"
            | "dedup"
            | "dedup_by"
            | "dedup_by_key"
            | "select_nth_unstable"
            | "select_nth_unstable_by" => {
                self.has_algorithms = true;
            }
            _ => {}
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.has_loops = true;
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.has_loops = true;
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        self.has_loops = true;
        syn::visit::visit_expr_loop(self, expr);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        self.has_branch_on_data = true;
        syn::visit::visit_expr_if(self, expr);
    }

    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        self.has_branch_on_data = true;
        syn::visit::visit_expr_match(self, expr);
    }
}

/// Check if signature inputs/outputs and body indicate host data-processing / simulation work.
pub(super) fn is_data_processing_ast(sig: &syn::Signature, block: &syn::Block) -> bool {
    let has_inputs = has_data_input_ast(sig);

    let has_typed_inputs = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, syn::FnArg::Typed(_)));
    let has_receiver = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)));
    if (has_typed_inputs && !has_inputs) || (!has_typed_inputs && has_receiver) {
        return false;
    }

    let mut visitor = BodyFeatureVisitor::default();
    visitor.visit_block(block);

    // If function is a byte codec operating over byte buffers &[u8] / &mut [u8]
    // and only performs bounds checks, slicing, and byte copying without semantic arithmetic/numeric transforms:
    if has_byte_buffer_param(sig) {
        if !visitor.has_payload_arithmetic
            && !visitor.has_numeric_method
            && !visitor.has_iterators_or_transforms
            && !visitor.has_algorithms
        {
            return false;
        }
    }

    visitor.has_semantic_operation()
}
