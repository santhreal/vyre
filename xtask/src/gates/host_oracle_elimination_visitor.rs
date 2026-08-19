//! `syn::visit::Visit` implementation for AST traversal and semantic data-flow tracking.

use std::collections::{BTreeMap, BTreeSet};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::gate::Finding;

use super::host_oracle_elimination_ast::AstAnalysisVisitor;
use super::host_oracle_elimination_classify::{
    has_data_output_ast, has_mutable_data_output_param, is_byte_unpack_codec_expr,
    is_data_processing_ast, is_dispatch_sizing_or_validator, is_fmt_signature, is_wire_codec_ast,
    returns_result_unit, returns_unit, BodyFeatureVisitor,
};
use super::host_oracle_elimination_extract::{extract_read_idents_from_expr, is_pure_decoder_loop};
use super::host_oracle_elimination_records::{
    compares_operands, computes_from_operands, extract_use_tree, CallSiteRecord, FunctionRecord,
    StaticConstRecord, FIX, SCALAR_TYPES,
};
use super::host_oracle_elimination_scanners::{
    compute_known_dispatch_exec_fns, FileImportCollector,
};
use crate::gates::scan::attribute_is_test_only;

impl<'ast> Visit<'ast> for AstAnalysisVisitor {
    fn visit_file(&mut self, file: &'ast syn::File) {
        self.scope_imports = vec![BTreeMap::new()];
        let mut collector = FileImportCollector {
            scope_imports: &mut self.scope_imports,
        };
        collector.visit_file(file);
        let local_known = compute_known_dispatch_exec_fns(file, self);
        self.known_dispatch_exec_fns.extend(local_known);
        syn::visit::visit_file(self, file);
    }
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if let Some(current_scope) = self.scope_imports.last_mut() {
            extract_use_tree("", &item.tree, current_scope);
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        let ident = item.ident.to_string();
        let mut qualified_parts = vec!["crate".to_string()];
        qualified_parts.extend(self.current_module.clone());
        qualified_parts.push(ident.clone());
        let qualified_path = qualified_parts.join("::");

        let has_pub_field = item
            .fields
            .iter()
            .any(|f| matches!(f.vis, syn::Visibility::Public(_)));
        if has_pub_field {
            self.types_with_public_fields.insert(qualified_path);
        }

        if self.struct_contains_canonical_dispatcher(item) {
            self.struct_types_with_dispatcher
                .insert(self.qualified_local_type_name(&item.ident));
        }

        syn::visit::visit_item_struct(self, item);
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        let ident = item.ident.to_string();
        let mut qualified_parts = vec!["crate".to_string()];
        qualified_parts.extend(self.current_module.clone());
        qualified_parts.push(ident);
        let qualified_path = qualified_parts.join("::");

        let has_pub_field = item
            .fields
            .named
            .iter()
            .any(|f| matches!(f.vis, syn::Visibility::Public(_)));
        if has_pub_field {
            self.types_with_public_fields.insert(qualified_path);
        }

        syn::visit::visit_item_union(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let mod_name = item.ident.to_string();
        let is_test_mod = item.attrs.iter().any(attribute_is_test_only);

        self.current_module.push(mod_name);
        self.scope_imports.push(BTreeMap::new());
        if is_test_mod {
            self.test_mod_depth += 1;
        }

        syn::visit::visit_item_mod(self, item);

        if is_test_mod {
            self.test_mod_depth -= 1;
        }
        self.scope_imports.pop();
        self.current_module.pop();
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let is_test_impl = item.attrs.iter().any(attribute_is_test_only);
        if is_test_impl {
            self.test_impl_depth += 1;
        }

        let self_has_dispatcher = self.type_contains_canonical_dispatcher(&item.self_ty);
        let prev_self_is_dispatcher = self.current_impl_self_is_dispatcher;
        if self_has_dispatcher {
            self.current_impl_self_is_dispatcher = true;
        }
        let is_fmt_trait = if let Some((_, trait_path, _)) = &item.trait_ {
            let trait_str = self.resolve_path_str(trait_path);
            trait_str == "std::fmt::Display"
                || trait_str == "core::fmt::Display"
                || trait_str == "Display"
                || trait_str == "std::fmt::Debug"
                || trait_str == "core::fmt::Debug"
                || trait_str == "Debug"
                || trait_str == "std::error::Error"
                || trait_str == "core::error::Error"
                || trait_str == "Error"
        } else {
            false
        };
        let is_std_trait = if let Some((_, trait_path, _)) = &item.trait_ {
            let last = trait_path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            matches!(
                last.as_str(),
                "Default"
                    | "Clone"
                    | "FromIterator"
                    | "PartialEq"
                    | "Eq"
                    | "PartialOrd"
                    | "Ord"
                    | "Hash"
                    | "Drop"
                    | "From"
                    | "Into"
                    | "TryFrom"
                    | "TryInto"
                    | "AsRef"
                    | "AsMut"
                    | "Borrow"
                    | "BorrowMut"
                    | "Deref"
                    | "DerefMut"
                    | "Iterator"
                    | "ExactSizeIterator"
                    | "DoubleEndedIterator"
                    | "Extend"
            )
        } else {
            false
        };

        if is_fmt_trait {
            self.fmt_impl_depth += 1;
        }
        if is_std_trait {
            self.trait_impl_depth += 1;
        }

        syn::visit::visit_item_impl(self, item);

        if is_std_trait {
            self.trait_impl_depth -= 1;
        }
        if is_fmt_trait {
            self.fmt_impl_depth -= 1;
        }

        if is_test_impl {
            self.test_impl_depth -= 1;
        }
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        let is_test_trait = item.attrs.iter().any(attribute_is_test_only);
        if is_test_trait {
            self.test_trait_depth += 1;
        }

        syn::visit::visit_item_trait(self, item);

        if is_test_trait {
            self.test_trait_depth -= 1;
        }
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        let is_test = item.attrs.iter().any(attribute_is_test_only);
        if is_test {
            self.item_test_depth += 1;
        }
        let const_name = item.ident.to_string();
        let line = item.ident.span().start().line as u32;

        let mut feature_visitor = BodyFeatureVisitor::default();
        feature_visitor.visit_expr(&item.expr);
        let has_semantic = feature_visitor.has_semantic_operation();

        self.static_consts.push(StaticConstRecord {
            name: const_name,
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped: self.in_test(),
            has_semantic_operation: has_semantic,
        });

        syn::visit::visit_item_const(self, item);

        if is_test {
            self.item_test_depth -= 1;
        }
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        let is_test = item.attrs.iter().any(attribute_is_test_only);
        if is_test {
            self.item_test_depth += 1;
        }
        let static_name = item.ident.to_string();
        let line = item.ident.span().start().line as u32;

        let mut feature_visitor = BodyFeatureVisitor::default();
        feature_visitor.visit_expr(&item.expr);
        let has_semantic = feature_visitor.has_semantic_operation();

        self.static_consts.push(StaticConstRecord {
            name: static_name,
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped: self.in_test(),
            has_semantic_operation: has_semantic,
        });

        syn::visit::visit_item_static(self, item);

        if is_test {
            self.item_test_depth -= 1;
        }
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        let mac_name = item
            .mac
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let is_op_submittal_macro =
            mac_name.starts_with("submit_") || mac_name.contains("intrinsic");
        let mut parsed = false;
        if let Ok(file) = syn::parse2::<syn::File>(item.mac.tokens.clone()) {
            parsed = true;
            for it in &file.items {
                self.visit_item(it);
            }
        } else if let Ok(expr) = syn::parse2::<syn::Expr>(item.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&expr);
        } else if let Ok(stmt) = syn::parse2::<syn::Stmt>(item.mac.tokens.clone()) {
            parsed = true;
            self.visit_stmt(&stmt);
        }
        if !parsed {
            if is_op_submittal_macro {
                self.in_op_reg_depth += 1;
                self.scan_macro_tokens_for_references(&item.mac.tokens);
                self.in_op_reg_depth -= 1;
            } else {
                self.scan_macro_tokens_for_references(&item.mac.tokens);
            }
        }
    }

    fn visit_expr_macro(&mut self, expr: &'ast syn::ExprMacro) {
        let mut parsed = false;
        if let Ok(inner_expr) = syn::parse2::<syn::Expr>(expr.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&inner_expr);
        } else if let Ok(punctuated) =
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(expr.mac.tokens.clone())
        {
            parsed = true;
            for inner in &punctuated {
                self.visit_expr(inner);
            }
        }
        if !parsed {
            self.scan_macro_tokens_for_references(&expr.mac.tokens);
        }
    }

    fn visit_stmt_macro(&mut self, stmt: &'ast syn::StmtMacro) {
        let mut parsed = false;
        if let Ok(inner_expr) = syn::parse2::<syn::Expr>(stmt.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&inner_expr);
        } else if let Ok(punctuated) =
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(stmt.mac.tokens.clone())
        {
            parsed = true;
            for inner in &punctuated {
                self.visit_expr(inner);
            }
        }
        if !parsed {
            self.scan_macro_tokens_for_references(&stmt.mac.tokens);
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(attribute_is_test_only);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = self.is_gpu_dispatch_sig(&item.sig, &item.block);
        let returns_data =
            has_data_output_ast(&item.sig) || has_mutable_data_output_param(&item.sig);
        let is_explicit = fn_name == "cpu_ref"
            || fn_name == "cpu_reference"
            || fn_name.starts_with("cpu_ref_")
            || fn_name.ends_with("_cpu_ref");

        let is_public = matches!(item.vis, syn::Visibility::Public(_));
        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry =
            returns_unit(&item.sig.output) && !has_mutable_data_output_param(&item.sig);
        let is_pure_validator = (returns_result_unit(&item.sig.output)
            || is_dispatch_sizing_or_validator(&item.sig)
            || fn_name.starts_with("validate_")
            || fn_name.starts_with("expect_"))
            && !has_mutable_data_output_param(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = is_wire_codec_ast(&item.sig, &item.block);

        let is_const = item.sig.constness.is_some();
        let is_sizing_or_validator = is_dispatch_sizing_or_validator(&item.sig);
        let is_dp = is_explicit
            || (!is_const
                && !is_fmt_method
                && !is_pure_telemetry
                && !is_pure_validator
                && !is_sizing_or_validator
                && !is_ir_inspector
                && !is_metadata_inspector
                && !is_wire_codec
                && !is_ir
                && returns_data
                && is_data_processing_ast(&item.sig, &item.block));
        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, Some(&item.block));
        let stages_semantic_resident_upload =
            self.stages_semantic_resident_upload(&item.block.stmts);

        let has_collection_payload_inputs = item.sig.inputs.iter().any(|arg| match arg {
            syn::FnArg::Typed(pat) => match &*pat.ty {
                syn::Type::Slice(_) => true,
                syn::Type::Reference(r) => match &*r.elem {
                    syn::Type::Slice(_) => true,
                    syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                        s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                    }),
                    _ => false,
                },
                syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                    s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                }),
                _ => false,
            },
            _ => false,
        });
        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_public,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            is_sizing_or_validator,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            has_collection_payload_inputs,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        let prev_fn = self.current_fn_idx.replace(fn_idx);
        let is_expected_output = fn_name == "expected_output"
            || fn_name.starts_with("expected_output_")
            || fn_name.ends_with("_expected_output");
        if is_expected_output {
            self.in_expected_output_depth += 1;
        }
        if is_fn_test_attr {
            self.item_test_depth += 1;
        }

        self.inspect_function_statements(&item.block.stmts, is_gpu);

        if is_fn_test_attr {
            self.item_test_depth -= 1;
        }
        if is_expected_output {
            self.in_expected_output_depth -= 1;
        }
        self.current_fn_idx = prev_fn;
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(attribute_is_test_only);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = self.is_gpu_dispatch_sig(&item.sig, &item.block);
        let returns_data =
            has_data_output_ast(&item.sig) || has_mutable_data_output_param(&item.sig);
        let is_explicit = fn_name == "cpu_ref" || fn_name == "cpu_reference";

        let is_public = matches!(item.vis, syn::Visibility::Public(_));
        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry =
            returns_unit(&item.sig.output) && !has_mutable_data_output_param(&item.sig);
        let is_trait_impl_method = self.trait_impl_depth > 0;
        let is_pure_validator = (returns_result_unit(&item.sig.output)
            || is_dispatch_sizing_or_validator(&item.sig)
            || is_trait_impl_method
            || fn_name.starts_with("validate_")
            || fn_name.starts_with("expect_"))
            && !has_mutable_data_output_param(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = is_wire_codec_ast(&item.sig, &item.block);
        let is_const = item.sig.constness.is_some();
        let is_sizing_or_validator = is_dispatch_sizing_or_validator(&item.sig);
        let is_dp = is_explicit
            || (!is_const
                && !is_fmt_method
                && !is_trait_impl_method
                && !is_pure_telemetry
                && !is_pure_validator
                && !is_sizing_or_validator
                && !is_ir_inspector
                && !is_metadata_inspector
                && !is_wire_codec
                && !is_ir
                && returns_data
                && is_data_processing_ast(&item.sig, &item.block));
        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, Some(&item.block));
        let stages_semantic_resident_upload =
            self.stages_semantic_resident_upload(&item.block.stmts);

        let has_collection_payload_inputs = item.sig.inputs.iter().any(|arg| match arg {
            syn::FnArg::Typed(pat) => match &*pat.ty {
                syn::Type::Slice(_) => true,
                syn::Type::Reference(r) => match &*r.elem {
                    syn::Type::Slice(_) => true,
                    syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                        s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                    }),
                    _ => false,
                },
                syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                    s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                }),
                _ => false,
            },
            _ => false,
        });
        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_public,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            is_sizing_or_validator,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            has_collection_payload_inputs,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        let prev_fn = self.current_fn_idx.replace(fn_idx);
        let is_expected_output = fn_name == "expected_output"
            || fn_name.starts_with("expected_output_")
            || fn_name.ends_with("_expected_output");
        if is_expected_output {
            self.in_expected_output_depth += 1;
        }
        if is_fn_test_attr {
            self.item_test_depth += 1;
        }

        self.inspect_function_statements(&item.block.stmts, is_gpu);

        if is_fn_test_attr {
            self.item_test_depth -= 1;
        }
        if is_expected_output {
            self.in_expected_output_depth -= 1;
        }
        self.current_fn_idx = prev_fn;
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(attribute_is_test_only);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = if let Some(block) = &item.default {
            self.is_gpu_dispatch_sig(&item.sig, block)
        } else {
            false
        };
        let returns_data =
            has_data_output_ast(&item.sig) || has_mutable_data_output_param(&item.sig);
        let is_explicit = fn_name == "cpu_ref" || fn_name == "cpu_reference";

        let is_public = true;
        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry =
            returns_unit(&item.sig.output) && !has_mutable_data_output_param(&item.sig);
        let is_pure_validator = (returns_result_unit(&item.sig.output)
            || is_dispatch_sizing_or_validator(&item.sig)
            || fn_name.starts_with("validate_")
            || fn_name.starts_with("expect_"))
            && !has_mutable_data_output_param(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = item
            .default
            .as_ref()
            .is_some_and(|block| is_wire_codec_ast(&item.sig, block));

        let is_const = item.sig.constness.is_some();
        let is_sizing_or_validator = is_dispatch_sizing_or_validator(&item.sig);
        let is_dp = is_explicit
            || (if let Some(block) = &item.default {
                !is_const
                    && !is_fmt_method
                    && !is_pure_telemetry
                    && !is_pure_validator
                    && !is_sizing_or_validator
                    && !is_ir_inspector
                    && !is_metadata_inspector
                    && !is_wire_codec
                    && !is_ir
                    && returns_data
                    && is_data_processing_ast(&item.sig, block)
            } else {
                false
            });
        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, item.default.as_ref());
        let stages_semantic_resident_upload = item
            .default
            .as_ref()
            .is_some_and(|block| self.stages_semantic_resident_upload(&block.stmts));

        let has_collection_payload_inputs = item.sig.inputs.iter().any(|arg| match arg {
            syn::FnArg::Typed(pat) => match &*pat.ty {
                syn::Type::Slice(_) => true,
                syn::Type::Reference(r) => match &*r.elem {
                    syn::Type::Slice(_) => true,
                    syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                        s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                    }),
                    _ => false,
                },
                syn::Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                    s.ident == "Vec" || s.ident == "BTreeSet" || s.ident == "HashSet"
                }),
                _ => false,
            },
            _ => false,
        });
        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_public,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            is_sizing_or_validator,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            has_collection_payload_inputs,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        if let Some(block) = &item.default {
            let prev_fn = self.current_fn_idx.replace(fn_idx);
            let is_expected_output = fn_name == "expected_output"
                || fn_name.starts_with("expected_output_")
                || fn_name.ends_with("_expected_output");
            if is_expected_output {
                self.in_expected_output_depth += 1;
            }
            if is_fn_test_attr {
                self.item_test_depth += 1;
            }
            self.inspect_function_statements(&block.stmts, is_gpu);
            if is_fn_test_attr {
                self.item_test_depth -= 1;
            }
            if is_expected_output {
                self.in_expected_output_depth -= 1;
            }
            self.current_fn_idx = prev_fn;
        }
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        let expr_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&expr.expr);
        syn::visit::visit_expr(self, &expr.expr);
        if expr_is_dispatch {
            self.post_dispatch_phase = true;
        }

        for arm in &expr.arms {
            let pat_str = quote::quote!(#arm.pat).to_string().replace(' ', "");
            let is_fallback_arm = self.in_gpu_dispatch_root
                && (pat_str.starts_with("Err(")
                    || pat_str.starts_with("Result::Err(")
                    || pat_str == "None"
                    || pat_str == "Option::None"
                    || pat_str == "false");

            if is_fallback_arm {
                self.in_fallback_depth += 1;
            }
            if let Some((_, guard)) = &arm.guard {
                syn::visit::visit_expr(self, guard);
            }
            syn::visit::visit_expr(self, &arm.body);
            if is_fallback_arm {
                self.in_fallback_depth -= 1;
            }
        }
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        let cond_str = quote::quote!(#expr.cond).to_string().replace(' ', "");
        let is_fallback_cond = self.in_gpu_dispatch_root
            && (cond_str.contains(".is_err()")
                || cond_str.contains("!*.is_ok()")
                || cond_str.contains(".is_none()")
                || cond_str.starts_with("letErr(")
                || cond_str.starts_with("letResult::Err(")
                || cond_str.starts_with("letNone="));

        let cond_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&expr.cond);
        syn::visit::visit_expr(self, &expr.cond);
        if cond_is_dispatch {
            self.post_dispatch_phase = true;
        }

        if is_fallback_cond {
            self.in_fallback_depth += 1;
        }
        for stmt in &expr.then_branch.stmts {
            syn::visit::visit_stmt(self, stmt);
        }
        if is_fallback_cond {
            self.in_fallback_depth -= 1;
        }

        if let Some((_, else_branch)) = &expr.else_branch {
            syn::visit::visit_expr(self, else_branch);
        }
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth > 0
        {
            let line = expr.span().start().line as u32;
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                "OperationRegistration `expected_output` defines dynamic helper closure alias; \
                 expected outputs must be exact byte literal constants"
                    .to_string(),
                FIX,
            ));
        }

        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth == 0
        {
            let mut feature_visitor = BodyFeatureVisitor::default();
            feature_visitor.visit_expr(&expr.body);
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline expected_output closure at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_public: false,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    is_sizing_or_validator: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    has_collection_payload_inputs: false,
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_method_call: false,
                    is_in_test: false,
                    is_in_expected_output: true,
                    is_in_fallback: false,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        if self.in_fallback_depth > 0 && !self.in_test() && self.in_synthetic_oracle_depth == 0 {
            let mut feature_visitor = BodyFeatureVisitor::default();
            feature_visitor.visit_expr(&expr.body);
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline fallback closure at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_public: false,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    is_sizing_or_validator: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    has_collection_payload_inputs: false,
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_method_call: false,
                    is_in_test: false,
                    is_in_expected_output: false,
                    is_in_fallback: true,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        // A closure body is host code the enclosing dispatch root does not
        // execute, except when the closure is the argument of a combinator on the
        // dispatch result: there the body is the post-dispatch phase itself.
        let keeps_dispatch_phase = self.in_post_dispatch_combinator_depth > 0;
        let prev_in_gpu = self.in_gpu_dispatch_root;
        let prev_post_dispatch = self.post_dispatch_phase;
        if !keeps_dispatch_phase {
            self.in_gpu_dispatch_root = false;
            self.post_dispatch_phase = false;
        }
        self.in_synthetic_oracle_depth += 1;
        syn::visit::visit_expr_closure(self, expr);
        self.in_synthetic_oracle_depth -= 1;
        self.in_gpu_dispatch_root = prev_in_gpu;
        self.post_dispatch_phase = prev_post_dispatch;
    }

    fn visit_expr_block(&mut self, expr: &'ast syn::ExprBlock) {
        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth == 0
        {
            let mut feature_visitor = BodyFeatureVisitor::default();
            for stmt in &expr.block.stmts {
                feature_visitor.visit_stmt(stmt);
            }
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline expected_output block at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_public: false,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    is_sizing_or_validator: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    has_collection_payload_inputs: false,
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_method_call: false,
                    is_in_test: false,
                    is_in_expected_output: true,
                    is_in_fallback: false,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }
        if self.in_fallback_depth > 0 && !self.in_test() && self.in_synthetic_oracle_depth == 0 {
            let mut feature_visitor = BodyFeatureVisitor::default();
            for stmt in &expr.block.stmts {
                feature_visitor.visit_stmt(stmt);
            }
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline fallback block at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_public: false,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    is_sizing_or_validator: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    has_collection_payload_inputs: false,
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_method_call: false,
                    is_in_test: false,
                    is_in_expected_output: false,
                    is_in_fallback: true,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }
        self.in_synthetic_oracle_depth += 1;
        syn::visit::visit_expr_block(self, expr);
        self.in_synthetic_oracle_depth -= 1;
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        let contains_dispatch = self.is_dispatch_execution_expr(&expr.expr)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !is_pure_decoder_loop(expr)
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        let contains_dispatch = self.is_dispatch_execution_expr(&expr.cond)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        let contains_dispatch = expr
            .body
            .stmts
            .iter()
            .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_loop(self, expr);
    }
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if self.in_gpu_dispatch_root && self.post_dispatch_phase && !self.in_test() {
            let reads_operands_as_data =
                computes_from_operands(&expr.op) || compares_operands(&expr.op);
            if reads_operands_as_data {
                let line = expr.span().start().line as u32;
                let is_dispatcher_call =
                    self.is_dispatch_execution_expr(&syn::Expr::Binary(expr.clone()));
                if !is_dispatcher_call {
                    let is_permitted_codec = is_byte_unpack_codec_expr(expr);
                    if !is_permitted_codec {
                        let operates_on_dispatched = if !self.dispatched_data_vars.is_empty() {
                            let reads =
                                extract_read_idents_from_expr(&syn::Expr::Binary(expr.clone()));
                            reads
                                .iter()
                                .any(|id| self.dispatched_data_vars.contains(id))
                        } else {
                            false
                        };
                        if operates_on_dispatched {
                            let current_fn_name = self
                                .current_fn_idx
                                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                                .unwrap_or_else(|| "<anonymous>".to_string());
                            self.direct_findings.push(Finding::at(
                                self.file.clone(),
                                line,
                                format!(
                                    "GPU dispatch function `{current_fn_name}` executes post-dispatch host arithmetic / semantic derivation on GPU results; \
                                     mathematical operations on dispatch output must be dispatched on GPU"
                                ),
                                FIX,
                            ));
                        }
                    }
                }
            }
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        let (full_callee, resolved_callee) = if let syn::Expr::Path(path_expr) = &*expr.func {
            let raw_callee = quote::quote!(#path_expr).to_string().replace(' ', "");
            let resolved = self
                .resolve_qualified_fn_path(&path_expr.path)
                .unwrap_or_else(|| self.resolve_path_str(&path_expr.path));
            let line = expr.func.span().start().line as u32;
            self.calls.push(CallSiteRecord {
                callee: resolved.clone(),
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_method_call: false,
                is_in_test: self.in_test(),
                is_in_expected_output: self.in_expected_output_depth > 0,
                is_in_fallback: self.in_fallback_depth > 0,
                is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
                is_in_op_reg: self.in_op_reg_depth > 0,
            });
            (raw_callee, resolved)
        } else {
            (String::new(), String::new())
        };

        // Structurally inspect OperationRegistration constructor calls to mark expected_output fixture contexts
        // Requires exact canonical origin vyre_foundation::operation::OperationRegistration
        let expected_idx = match resolved_callee.as_str() {
            "vyre_foundation::operation::OperationRegistration::library" => Some(3),
            "vyre_foundation::operation::OperationRegistration::primitive" => Some(3),
            "vyre_foundation::operation::OperationRegistration::intrinsic" => Some(4),
            "vyre_foundation::operation::OperationRegistration::new" => Some(4),
            _ => None,
        };

        if let Some(target_idx) = expected_idx {
            self.in_op_reg_depth += 1;
            syn::visit::visit_expr(self, &expr.func);
            for (idx, arg) in expr.args.iter().enumerate() {
                if idx == target_idx {
                    self.in_expected_output_depth += 1;
                    syn::visit::visit_expr(self, arg);
                    self.in_expected_output_depth -= 1;
                } else {
                    syn::visit::visit_expr(self, arg);
                }
            }
            self.in_op_reg_depth -= 1;
            return;
        }

        syn::visit::visit_expr_call(self, expr);
        if self.in_expected_output_depth > 0 && !self.in_test() {
            let callee_ident = if let syn::Expr::Path(path_expr) = &*expr.func {
                path_expr.path.segments.last().map(|s| s.ident.to_string())
            } else {
                None
            };
            let is_permitted_constructor = match callee_ident.as_deref() {
                Some("vec") | Some("Vec") | Some("Some") | Some("None") | Some("Ok") => true,
                _ => false,
            };
            if !is_permitted_constructor {
                let line = expr.func.span().start().line as u32;
                self.direct_findings.push(Finding::at(
                    self.file.clone(),
                    line,
                    format!(
                        "OperationRegistration `expected_output` invokes dynamic helper/codec `{full_callee}`; \
                         expected outputs must be exact byte literal constants"
                    ),
                    FIX,
                ));
            }
        }
    }

    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        let resolved_struct = self.resolve_path_str(&expr.path);
        let is_op_reg = resolved_struct == "vyre_foundation::operation::OperationRegistration";

        if is_op_reg {
            self.in_op_reg_depth += 1;
        }
        for field in &expr.fields {
            let is_expected = is_op_reg
                && match &field.member {
                    syn::Member::Named(ident) => ident == "expected_output",
                    _ => false,
                };

            if is_expected {
                self.in_expected_output_depth += 1;
                syn::visit::visit_expr(self, &field.expr);
                self.in_expected_output_depth -= 1;
            } else {
                syn::visit::visit_expr(self, &field.expr);
            }
        }
        if let Some(rest) = &expr.rest {
            syn::visit::visit_expr(self, rest);
        }
        if is_op_reg {
            self.in_op_reg_depth -= 1;
        }
    }

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        let path_str = quote::quote!(#expr).to_string().replace(' ', "");
        let line = expr.span().start().line as u32;

        if self.in_op_reg_depth > 0 && self.in_expected_output_depth == 0 && !self.in_test() {
            // Function passed by name to OperationRegistration (e.g. Some(generate_deterministic_inputs), add_program)
            self.calls.push(CallSiteRecord {
                callee: path_str.clone(),
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_method_call: false,
                is_in_test: false,
                is_in_expected_output: false,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                is_in_op_reg: true,
            });
        }

        syn::visit::visit_expr_path(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method_name = expr.method.to_string();
        let line = expr.method.span().start().line as u32;

        self.calls.push(CallSiteRecord {
            callee: method_name.clone(),
            caller_file: self.file.clone(),
            caller_module: self.current_module.clone(),
            caller_fn_idx: self.current_fn_idx,
            line,
            is_method_call: true,
            is_in_test: self.in_test(),
            is_in_expected_output: self.in_expected_output_depth > 0,
            is_in_fallback: self.in_fallback_depth > 0,
            is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
            is_in_op_reg: self.in_op_reg_depth > 0,
        });

        if self.in_expected_output_depth > 0 && !self.in_test() {
            let is_permitted_alloc = method_name == "to_vec" || method_name == "clone";
            if !is_permitted_alloc {
                self.direct_findings.push(Finding::at(
                    self.file.clone(),
                    line,
                    format!(
                        "OperationRegistration `expected_output` invokes dynamic method `.{method_name}`; \
                         expected outputs must be exact byte literal constants"
                    ),
                    FIX,
                ));
            }
        }

        // Check if receiver executes dispatch (e.g. dispatcher.dispatch(..).map(..))
        let receiver_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&expr.receiver);

        syn::visit::visit_expr(self, &expr.receiver);
        if receiver_is_dispatch {
            self.post_dispatch_phase = true;
        }

        // Convict post-dispatch host reduction / semantic methods in shipping dispatcher functions
        if self.in_gpu_dispatch_root && self.post_dispatch_phase && !self.in_test() {
            let is_dispatcher_call =
                self.is_dispatch_execution_expr(&syn::Expr::MethodCall(expr.clone()));
            if !is_dispatcher_call {
                match method_name.as_str() {
                    "any" | "all" | "sum" | "fold" | "rfold" | "reduce" | "count"
                    | "count_ones" | "count_zeros" | "leading_zeros" | "trailing_zeros"
                    | "rotate_left" | "rotate_right" | "reverse_bits" | "wrapping_add"
                    | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "checked_add" => {
                        let receiver_idents = extract_read_idents_from_expr(&expr.receiver);
                        let receiver_is_diagnostic = receiver_idents
                            .iter()
                            .any(|id| self.non_data_diagnostic_params.contains(id));
                        let operates_on_dispatched = receiver_is_dispatch
                            || receiver_idents
                                .iter()
                                .any(|id| self.dispatched_data_vars.contains(id))
                            || !receiver_is_diagnostic;
                        if operates_on_dispatched {
                            let current_fn_name = self
                                .current_fn_idx
                                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                                .unwrap_or_else(|| "<anonymous>".to_string());
                            self.direct_findings.push(Finding::at(
                                self.file.clone(),
                                line,
                                format!(
                                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host reduction/aggregation `.{method_name}`; \
                                     post-dispatch semantic reductions must be dispatched on GPU"
                                ),
                                FIX,
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        let is_fallback_combinator = (self.in_gpu_dispatch_root
            || self.is_dispatch_execution_expr(&expr.receiver))
            && (method_name == "unwrap_or_else"
                || method_name == "or_else"
                || method_name == "unwrap_or"
                || method_name == "unwrap_or_default");
        // A combinator chained onto the dispatch result runs its argument on the
        // host with that result in hand, so `dispatch(..).map(|out| out.iter()
        // .count())` is the same host reduction as the statement form. A fallback
        // combinator keeps its own context, which the fallback rules own.
        let carries_dispatch_result = receiver_is_dispatch && !is_fallback_combinator;
        for arg in &expr.args {
            if is_fallback_combinator {
                self.in_fallback_depth += 1;
                syn::visit::visit_expr(self, arg);
                self.in_fallback_depth -= 1;
            } else if method_name == "with_expected_output" {
                self.in_expected_output_depth += 1;
                syn::visit::visit_expr(self, arg);
                self.in_expected_output_depth -= 1;
            } else if carries_dispatch_result {
                self.in_post_dispatch_combinator_depth += 1;
                syn::visit::visit_expr(self, arg);
                self.in_post_dispatch_combinator_depth -= 1;
            } else {
                syn::visit::visit_expr(self, arg);
            }
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let path_str = quote::quote!(#path).to_string().replace(' ', "");
        let line = path.span().start().line as u32;

        if path_str.contains("vyre_reference") || path_str.contains("SubgroupSimulator") {
            self.calls.push(CallSiteRecord {
                callee: "vyre_reference".to_string(),
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_method_call: false,
                is_in_test: self.in_test(),
                is_in_expected_output: self.in_expected_output_depth > 0,
                is_in_fallback: self.in_fallback_depth > 0,
                is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
                is_in_op_reg: self.in_op_reg_depth > 0,
            });
        }

        if self.in_expected_output_depth > 0 {
            let ident = path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            let is_primitive_or_std = SCALAR_TYPES.contains(&ident.as_str())
                || matches!(
                    ident.as_str(),
                    "Some"
                        | "None"
                        | "Ok"
                        | "Err"
                        | "Vec"
                        | "vec"
                        | "Option"
                        | "Result"
                        | "Arc"
                        | "Box"
                        | "Rc"
                        | "Cell"
                        | "RefCell"
                        | "str"
                        | "String"
                );
            if !is_primitive_or_std {
                let resolved = self
                    .resolve_qualified_fn_path(path)
                    .unwrap_or_else(|| self.resolve_path_str(path));
                self.calls.push(CallSiteRecord {
                    callee: resolved,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: self.current_fn_idx,
                    line,
                    is_method_call: false,
                    is_in_test: self.in_test(),
                    is_in_expected_output: true,
                    is_in_fallback: false,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        syn::visit::visit_path(self, path);
    }
}
