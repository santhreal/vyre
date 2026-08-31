//! Source analysis and semantic rule evaluation for host oracle elimination gate.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use syn::visit::Visit;

use crate::gate::{Finding, GateError};
use crate::gates::scan::Tree;

use super::host_oracle_elimination_ast::AstAnalysisVisitor;
use super::host_oracle_elimination_records::{
    CallSiteRecord, FunctionRecord, StaticConstRecord, FIX,
};
use super::host_oracle_elimination_scanners::{
    compute_known_dispatch_exec_fns_multi, derive_canonical_dispatcher_methods,
    derive_canonical_execution_fns, derive_registration_expected_output_indices,
};

pub(super) fn analyze_sources(
    tree: &Tree,
    sources: &[PathBuf],
    test_scoped_files: &BTreeSet<PathBuf>,
) -> Result<Vec<Finding>, GateError> {
    let canonical_path = PathBuf::from("vyre-megakernel/src/execution.rs");
    let text = tree.read(&canonical_path).map_err(|err| {
        GateError::new(
            format!(
                "failed to read canonical SemanticExecutor source `{}`: {err}",
                canonical_path.display()
            ),
            "ensure canonical `vyre-megakernel/src/execution.rs` exists and is readable",
        )
    })?;

    let file_ast = syn::parse_file(&text).map_err(|err| {
        GateError::new(
            format!(
                "failed to parse canonical SemanticExecutor source `{}`: {err}",
                canonical_path.display()
            ),
            "fix syntax defect in SemanticExecutor trait definition",
        )
    })?;

    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_input_binding_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &file_ast,
        &mut canonical_trait_methods,
        &mut canonical_input_binding_methods,
    );

    let canonical_execution_fns =
        derive_canonical_execution_fns(&file_ast, &canonical_trait_methods);

    if canonical_trait_methods.is_empty()
        || canonical_input_binding_methods.is_empty()
        || canonical_execution_fns.is_empty()
    {
        return Err(GateError::new(
            format!(
                "canonical SemanticExecutor in `{}` yielded zero execution methods, input-binding methods, or free execution helpers",
                canonical_path.display()
            ),
            "ensure the SemanticExecutor trait defines an execution method taking a SemanticExecutionRequest and returning Result, that SemanticExecutionRequest defines a constructor taking byte payloads, and that at least one free helper takes the executor and calls that method",
        ));
    }

    let registration_path = PathBuf::from("vyre-foundation/src/operation/mod.rs");
    let registration_text = tree.read(&registration_path).map_err(|err| {
        GateError::new(
            format!(
                "failed to read operation registration source `{}`: {err}",
                registration_path.display()
            ),
            "ensure canonical `vyre-foundation/src/operation/mod.rs` exists and is readable",
        )
    })?;
    let registration_ast = syn::parse_file(&registration_text).map_err(|err| {
        GateError::new(
            format!(
                "failed to parse operation registration source `{}`: {err}",
                registration_path.display()
            ),
            "fix syntax defect in the OperationRegistration definition",
        )
    })?;
    let registration_expected_output_indices =
        derive_registration_expected_output_indices(&registration_ast);
    if registration_expected_output_indices.is_empty() {
        return Err(GateError::new(
            format!(
                "`impl OperationRegistration` in `{}` declares no constructor taking an `expected_output` argument",
                registration_path.display()
            ),
            "keep an OperationRegistration constructor whose expected-output callback parameter is named `expected_output`, so the gate can separate a reached fixture from a host oracle producing expected bytes",
        ));
    }
    let mut parsed_sources = Vec::new();
    for path in sources {
        let text = tree.read(path)?;
        let file_ast = syn::parse_file(&text).map_err(|err| {
            GateError::new(
                format!("failed to parse `{}`: {err}", path.display()),
                "fix syntax defect so the file parses as valid Rust",
            )
        })?;
        let is_test_scoped = test_scoped_files.contains(path);
        parsed_sources.push((path.clone(), file_ast, is_test_scoped));
    }
    Ok(analyze_parsed(
        &parsed_sources,
        &canonical_trait_methods,
        &canonical_input_binding_methods,
        &canonical_execution_fns,
        &registration_expected_output_indices,
    ))
}

/// Run the visitor over parsed sources and evaluate every rule over the union.
///
/// The gate parses the tree and its fixtures parse strings, and both analyze
/// through here. A second copy of this walk once sat in the fixtures, which is
/// the shape that lets a rule pass its fixture and never fire on the tree: the
/// copy is what the fixture proves, and nothing proves the copy still matches.
pub(super) fn analyze_parsed(
    parsed_sources: &[(PathBuf, syn::File, bool)],
    canonical_trait_methods: &BTreeSet<String>,
    canonical_input_binding_methods: &BTreeSet<String>,
    canonical_execution_fns: &BTreeSet<String>,
    registration_expected_output_indices: &BTreeMap<String, usize>,
) -> Vec<Finding> {
    let file_asts: Vec<(&Path, &syn::File)> = parsed_sources
        .iter()
        .map(|(path, ast, _)| (path.as_path(), ast))
        .collect();
    let global_known_dispatch_exec_fns = compute_known_dispatch_exec_fns_multi(
        &file_asts,
        canonical_trait_methods,
        canonical_input_binding_methods,
        canonical_execution_fns,
    );

    let mut all_functions = Vec::new();
    let mut all_calls = Vec::new();
    let mut all_static_consts = Vec::new();
    let mut all_findings = Vec::new();
    let mut all_types_with_public_fields = BTreeSet::new();
    for (path, file_ast, is_test_scoped) in parsed_sources {
        let fn_offset = all_functions.len();
        let mut visitor = AstAnalysisVisitor::new(
            path.clone(),
            *is_test_scoped,
            fn_offset,
            canonical_trait_methods.clone(),
            canonical_input_binding_methods.clone(),
        );
        visitor.known_dispatch_exec_fns = global_known_dispatch_exec_fns.clone();
        visitor.registration_expected_output_indices = registration_expected_output_indices.clone();
        // Pre-discover all types declared locally in this file
        for item in &file_ast.items {
            match item {
                syn::Item::Struct(s) => {
                    visitor.local_declared_types.insert(s.ident.to_string());
                }
                syn::Item::Enum(e) => {
                    visitor.local_declared_types.insert(e.ident.to_string());
                }
                syn::Item::Type(t) => {
                    visitor.local_declared_types.insert(t.ident.to_string());
                }
                syn::Item::Trait(tr) => {
                    visitor.local_declared_types.insert(tr.ident.to_string());
                }
                syn::Item::Union(u) => {
                    visitor.local_declared_types.insert(u.ident.to_string());
                }
                _ => {}
            }
        }

        visitor.visit_file(file_ast);

        all_functions.extend(visitor.functions);
        all_calls.extend(visitor.calls);
        all_static_consts.extend(visitor.static_consts);
        all_findings.extend(visitor.direct_findings);
        all_types_with_public_fields.extend(visitor.types_with_public_fields);
    }

    let evaluated = evaluate_rules(
        &all_functions,
        &all_calls,
        &all_static_consts,
        &all_types_with_public_fields,
    );
    all_findings.extend(evaluated);

    // Deduplicate findings by (file, line, message)
    let mut deduped_findings = Vec::new();
    let mut seen_findings = BTreeSet::new();
    for finding in all_findings {
        let key = (finding.file.clone(), finding.line, finding.message.clone());
        if seen_findings.insert(key) {
            deduped_findings.push(finding);
        }
    }
    deduped_findings
}

/// Evaluate zero-baseline host oracle and transitive reachability rules.
pub(super) fn evaluate_rules(
    functions: &[FunctionRecord],
    calls: &[CallSiteRecord],
    static_consts: &[StaticConstRecord],
    types_with_public_fields: &BTreeSet<String>,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Check direct vyre_reference calls in production code
    for call in calls {
        if !call.is_in_test && call.callee == "vyre_reference" {
            findings.push(Finding::at(
                call.caller_file.clone(),
                call.line,
                "production dependency or invocation of host simulator `vyre_reference`",
                FIX,
            ));
        }
    }

    // Build map from target name -> Vec<idx> for static_consts
    let mut static_const_indices_by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, sc) in static_consts.iter().enumerate() {
        if !sc.is_test_scoped {
            static_const_indices_by_target
                .entry(sc.name.clone())
                .or_default()
                .push(idx);
        }
    }

    // Build map from callee target name -> Vec<idx> for functions
    let mut func_indices_by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        if !func.is_test_scoped {
            func_indices_by_target
                .entry(func.name.clone())
                .or_default()
                .push(idx);
        }
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); functions.len()];
    let mut dynamic_expected_output_calls: BTreeSet<usize> = BTreeSet::new();
    let mut dynamic_fallback_calls: BTreeSet<usize> = BTreeSet::new();
    let mut dynamic_expected_output_static_consts: BTreeSet<usize> = BTreeSet::new();
    let mut reachable_from_roots = vec![false; functions.len()];
    let mut queue = VecDeque::new();

    for call in calls {
        if call.is_in_test {
            continue;
        }

        // Fail-closed function call resolution with turbofish and CFG-alternative support
        let mut matched_targets: Vec<usize> = Vec::new();
        let clean_callee = AstAnalysisVisitor::strip_turbofish(&call.callee);

        // 1. Check exact intra-module match (matches all compatible CFG definitions)
        for (target_idx, target_fn) in functions.iter().enumerate() {
            if target_fn.is_test_scoped {
                continue;
            }

            let is_exact_intra_module = call.caller_file == target_fn.file
                && call.caller_module == target_fn.module_path
                && (clean_callee == target_fn.name
                    || clean_callee.ends_with(&format!("::{}", target_fn.name)));

            if is_exact_intra_module {
                matched_targets.push(target_idx);
            }
        }

        // 2. If no exact intra-module match, check qualified path matches
        if matched_targets.is_empty() {
            for (target_idx, target_fn) in functions.iter().enumerate() {
                if target_fn.is_test_scoped {
                    continue;
                }

                let mod_path = target_fn.module_path.join("::");
                if !mod_path.is_empty()
                    && (clean_callee == format!("{mod_path}::{}", target_fn.name)
                        || clean_callee.ends_with(&format!("::{mod_path}::{}", target_fn.name)))
                {
                    matched_targets.push(target_idx);
                }
            }
        }

        // 3. If still unmatched, search by bare name across non-test targets (only for free fn calls, not common methods/constructors)
        let bare_name = clean_callee.rsplit("::").next().unwrap_or(clean_callee);
        let is_common_constructor_or_std_method = matches!(
            bare_name,
            "new"
                | "default"
                | "with_capacity"
                | "from"
                | "from_bytes"
                | "clone"
                | "into"
                | "as_ref"
                | "as_slice"
                | "len"
                | "is_empty"
                | "get"
                | "insert"
                | "push"
                | "pop"
                | "clear"
                | "load"
                | "and"
                | "chunks"
                | "chunks_exact"
                | "iter"
                | "as_bytes"
                | "to_vec"
                | "extend"
                | "split"
                | "trim"
                | "lines"
        );
        if matched_targets.is_empty()
            && !call.is_method_call
            && !is_common_constructor_or_std_method
        {
            if let Some(target_indices) = func_indices_by_target.get(bare_name) {
                matched_targets.extend(target_indices);
            } else {
                for (target_idx, target_fn) in functions.iter().enumerate() {
                    if !target_fn.is_test_scoped
                        && clean_callee.ends_with(&format!("::{}", target_fn.name))
                    {
                        matched_targets.push(target_idx);
                    }
                }
            }
        }
        for &target_idx in &matched_targets {
            if call.is_in_expected_output {
                dynamic_expected_output_calls.insert(target_idx);
            } else if call.is_in_fallback {
                dynamic_fallback_calls.insert(target_idx);
            } else if call.is_in_op_reg {
                // Top-level OperationRegistration arguments (build, test_inputs) are roots
                reachable_from_roots[target_idx] = true;
                queue.push_back(target_idx);
            } else if let Some(caller_idx) = call.caller_fn_idx {
                if caller_idx != target_idx {
                    adjacency[caller_idx].push(target_idx);
                }
            }
        }

        // Also resolve static/const references inside expected_output
        if call.is_in_expected_output {
            let mut matched_sc: Vec<usize> = Vec::new();

            for (sc_idx, sc) in static_consts.iter().enumerate() {
                if sc.is_test_scoped {
                    continue;
                }
                let is_exact = call.caller_file == sc.file
                    && call.caller_module == sc.module_path
                    && (call.callee == sc.name || call.callee.ends_with(&format!("::{}", sc.name)));
                if is_exact {
                    matched_sc.push(sc_idx);
                }
            }

            if matched_sc.is_empty() {
                for (sc_idx, sc) in static_consts.iter().enumerate() {
                    if sc.is_test_scoped {
                        continue;
                    }
                    let mod_path = sc.module_path.join("::");
                    if !mod_path.is_empty()
                        && call.callee.contains(&format!("{mod_path}::{}", sc.name))
                    {
                        matched_sc.push(sc_idx);
                    }
                }
            }

            if matched_sc.is_empty() {
                if let Some(sc_indices) = static_const_indices_by_target.get(&call.callee) {
                    matched_sc.extend(sc_indices);
                } else {
                    for (sc_idx, sc) in static_consts.iter().enumerate() {
                        if !sc.is_test_scoped && call.callee.ends_with(&format!("::{}", sc.name)) {
                            matched_sc.push(sc_idx);
                        }
                    }
                }
            }

            for &sc_idx in &matched_sc {
                dynamic_expected_output_static_consts.insert(sc_idx);
            }
        }
    }

    // Transitive propagation of expected_output callers to callees
    let mut dynamic_expected_output_queue: VecDeque<usize> =
        dynamic_expected_output_calls.iter().copied().collect();
    while let Some(curr) = dynamic_expected_output_queue.pop_front() {
        for &callee in &adjacency[curr] {
            if dynamic_expected_output_calls.insert(callee) {
                dynamic_expected_output_queue.push_back(callee);
            }
        }
    }

    // Flag referenced production static/const initializers with dynamic semantic operations
    for &sc_idx in &dynamic_expected_output_static_consts {
        let sc = &static_consts[sc_idx];
        if sc.has_semantic_operation {
            findings.push(Finding::at(
                sc.file.clone(),
                sc.line,
                format!(
                    "production operation registration `expected_output` references computed static/const `{}` containing dynamic semantic execution; \
                     operation registrations must use exact byte constants",
                    sc.name
                ),
                FIX,
            ));
        }
    }

    // Fixed-point propagation of GPU dispatch execution across function call graph
    let mut is_gpu_dispatch_exec: Vec<bool> = functions
        .iter()
        .map(|f| !f.is_test_scoped && f.is_gpu_dispatch_root)
        .collect();
    let mut dispatch_propagation_changed = true;
    while dispatch_propagation_changed {
        dispatch_propagation_changed = false;
        for (caller_idx, func) in functions.iter().enumerate() {
            if func.is_test_scoped
                || !func.has_canonical_dispatcher_param
                || is_gpu_dispatch_exec[caller_idx]
            {
                continue;
            }
            for &callee_idx in &adjacency[caller_idx] {
                if is_gpu_dispatch_exec[callee_idx] {
                    is_gpu_dispatch_exec[caller_idx] = true;
                    dispatch_propagation_changed = true;
                    break;
                }
            }
        }
    }

    // Seed production roots established strictly by canonical foundation types,
    // device dispatch calls, driver infrastructure, and operation registrations.
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }

        let is_driver_infra = func.file.to_string_lossy().contains("vyre-driver")
            && func.is_public
            && !func.is_explicit_oracle_name;
        let is_root = func.is_ir_builder || is_gpu_dispatch_exec[idx] || is_driver_infra;
        if is_root {
            reachable_from_roots[idx] = true;
            queue.push_back(idx);
        }
    }

    // Map each exact qualified return type to the list of non-test producer function indices
    let mut non_test_producers_by_type: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }
        for ret_ty in &func.return_custom_types {
            non_test_producers_by_type
                .entry(ret_ty.clone())
                .or_default()
                .push(idx);
        }
    }

    // Map function name to indices for resolving callee flows
    let mut fn_indices_by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        fn_indices_by_name
            .entry(func.name.clone())
            .or_default()
            .push(idx);
    }

    // Inter-procedural fixed-point propagation of parameter dispatch flow
    let mut param_dispatched: Vec<Vec<bool>> = functions
        .iter()
        .map(|f| {
            let mut disp = vec![false; f.params.len()];
            for &p_idx in &f.direct_dispatched_param_indices {
                if p_idx < disp.len() {
                    disp[p_idx] = true;
                }
            }
            disp
        })
        .collect();

    let mut param_flow_changed = true;
    while param_flow_changed {
        param_flow_changed = false;
        for (caller_idx, func) in functions.iter().enumerate() {
            for flow in &func.param_callee_flows {
                if flow.param_idx < param_dispatched[caller_idx].len()
                    && !param_dispatched[caller_idx][flow.param_idx]
                {
                    if let Some(callee_indices) = fn_indices_by_name.get(&flow.callee_name) {
                        for &callee_idx in callee_indices {
                            if flow.callee_arg_idx < param_dispatched[callee_idx].len()
                                && param_dispatched[callee_idx][flow.callee_arg_idx]
                            {
                                param_dispatched[caller_idx][flow.param_idx] = true;
                                param_flow_changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Map each function to the set of qualified custom types of parameters that flow into dispatch
    let func_dispatched_custom_types: Vec<BTreeSet<String>> = functions
        .iter()
        .enumerate()
        .map(|(fn_idx, func)| {
            let mut types = BTreeSet::new();
            for (p_idx, p_rec) in func.params.iter().enumerate() {
                if p_idx < param_dispatched[fn_idx].len() && param_dispatched[fn_idx][p_idx] {
                    types.extend(p_rec.qualified_custom_types.clone());
                }
            }
            types
        })
        .collect();

    // Fail-closed nominal bridge: a qualified custom type may root a producer only when:
    // (a) an actual call/dataflow path already connects it (handled by call graph BFS), OR
    // (b) it is the unique non-test producer of that exact qualified type AND a canonical
    //     dispatch-executing consumer accepts that exact qualified type AND structural
    //     dataflow proves the parameter feeds into dispatch execution handle_ids (direct
    //     or transitive) AND the type has no externally public fields that bypass the unique producer.
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped || !is_gpu_dispatch_exec[idx] {
            continue;
        }
        for param_ty in &func_dispatched_custom_types[idx] {
            if types_with_public_fields.contains(param_ty) {
                continue;
            }
            if let Some(producer_indices) = non_test_producers_by_type.get(param_ty) {
                if producer_indices.len() == 1 {
                    let p_idx = producer_indices[0];
                    if !reachable_from_roots[p_idx] {
                        reachable_from_roots[p_idx] = true;
                        queue.push_back(p_idx);
                    }
                }
            }
        }
    }

    // Fixed-point candidate propagation: functions returning scalar/collection output
    // that call a candidate also become candidates.
    let mut is_candidate: Vec<bool> = functions
        .iter()
        .map(|f| (f.is_explicit_oracle_name || f.is_data_processing) && !f.is_sizing_or_validator)
        .collect();
    let mut candidate_propagation_changed = true;
    while candidate_propagation_changed {
        candidate_propagation_changed = false;
        for (caller_idx, caller_fn) in functions.iter().enumerate() {
            if caller_fn.is_test_scoped
                || caller_fn.is_ir_builder
                || caller_fn.is_wire_codec
                || caller_fn.is_sizing_or_validator
                || is_candidate[caller_idx]
                || !caller_fn.returns_data_output
            {
                continue;
            }

            for &callee_idx in &adjacency[caller_idx] {
                let callee_fn = &functions[callee_idx];
                if is_candidate[callee_idx] && !callee_fn.is_sizing_or_validator {
                    is_candidate[caller_idx] = true;
                    candidate_propagation_changed = true;
                    break;
                }
            }
        }
    }

    // Transitive BFS traversal from production roots
    while let Some(curr) = queue.pop_front() {
        for &next in &adjacency[curr] {
            if !reachable_from_roots[next] {
                reachable_from_roots[next] = true;
                queue.push_back(next);
            }
        }
    }

    // Judge all declared non-test functions
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }

        // Rule 1: Explicit cpu_ref names in production code
        if func.is_explicit_oracle_name {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "production host reference oracle function definition `{}`",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 2: Expected output dynamic oracle call (unconditional on candidacy)
        if dynamic_expected_output_calls.contains(&idx) {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "production operation registration `expected_output` dynamically executes host oracle `{}`; \
                     operation registrations must use exact byte constants",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 3: Dispatch error / fallback oracle call
        if dynamic_fallback_calls.contains(&idx) && is_candidate[idx] {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "GPU dispatch function contains host CPU reference fallback `{}` on dispatch error/failure; \
                     silent host fallback is forbidden",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 4: Data-processing candidates must be transitively reachable from a production root
        if is_candidate[idx] && !reachable_from_roots[idx] {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "unisolated host data-processing semantic twin `{}` is not reachable from any production root; \
                     host semantic execution must live in tests or vyre-reference",
                    func.name
                ),
                FIX,
            ));
        }

        // Rule 5: GPU dispatch functions must not invoke host data-processing semantic helpers
        if is_gpu_dispatch_exec[idx] {
            for &callee_idx in &adjacency[idx] {
                if is_gpu_dispatch_exec[callee_idx] {
                    // Both caller and callee are verified SemanticExecutor/GPU dispatch wrappers
                    continue;
                }
                let callee = &functions[callee_idx];
                if callee.name == "build" && callee.is_ir_builder {
                    continue;
                }
                if is_candidate[callee_idx] {
                    // A staging helper is legitimate on either of two proofs: it
                    // binds a tainted payload into a canonical request itself, or
                    // the exact type it returns is a parameter of a canonical
                    // dispatch function that structural dataflow proves reaches
                    // submission. Binding is the submitting function's work at
                    // this seam, so requiring it of the producer would leave no
                    // legitimate staging shape at all.
                    let binds_payload_into_request = callee.stages_semantic_input_binding;
                    let carrier_reaches_submission =
                        callee.return_custom_types.iter().any(|return_type| {
                            func_dispatched_custom_types.iter().enumerate().any(
                                |(consumer_idx, dispatched_types)| {
                                    is_gpu_dispatch_exec[consumer_idx]
                                        && dispatched_types.contains(return_type)
                                },
                            )
                        });
                    let is_proven_resident_staging =
                        binds_payload_into_request || carrier_reaches_submission;
                    if is_proven_resident_staging {
                        continue;
                    }
                    if callee.has_collection_payload_inputs || callee.is_explicit_oracle_name {
                        findings.push(Finding::at(
                            func.file.clone(),
                            func.line,
                            format!(
                                "GPU dispatch function `{}` invokes host data-processing semantic helper `{}`; \
                                 host mathematical calculations must be executed on GPU",
                                func.name, callee.name
                            ),
                            FIX,
                        ));
                    }
                }
            }
        }
    }

    findings
}
