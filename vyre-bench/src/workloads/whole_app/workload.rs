//! Whole-application workload specification and its graph builder.
//!
//! A workload states the stages one representative application runs and builds
//! the program those stages compile to. Measuring it is the caller's step.

use super::*;

/// Whole-application workload specification and graph builder.
#[derive(Clone)]
pub struct WholeApplicationWorkload {
    /// Stable workload identifier.
    pub id: &'static str,
    /// Human-readable title.
    pub name: &'static str,
    /// Domain classification.
    pub domain: ApplicationDomain,
    /// Detailed description.
    pub description: &'static str,
    /// Pinned native baseline comparator identifier.
    pub pinned_native_baseline_id: &'static str,
    /// Pinned native baseline comparator name.
    pub pinned_native_baseline_name: &'static str,
    /// Default 14-point comparison equality conditions.
    pub default_conditions: NativeComparisonConditions,
    /// Builder constructing the connected multi-node ProgramGraph and concrete input buffers.
    pub build_graph_and_inputs: fn() -> Result<(ProgramGraph, BTreeMap<String, Vec<u8>>), String>,
}

impl WholeApplicationWorkload {
    /// Count internal dataflow edges connecting stages in the graph.
    #[must_use]
    pub fn count_internal_edges(graph: &ProgramGraph) -> usize {
        graph
            .values()
            .iter()
            .filter(|v| v.producer.is_some() && !v.consumers.is_empty())
            .count()
    }

    /// Validate that the constructed graph satisfies the whole-application topology
    /// requirement (at least 2 connected nodes and internal edges).
    pub fn validate_topology(&self, graph: &ProgramGraph) -> Result<(), WholeApplicationRefusal> {
        let node_count = graph.nodes().len();
        if node_count < 2 {
            return Err(WholeApplicationRefusal::SingleNodeIsolatedKernel {
                workload_id: self.id.to_string(),
                node_count,
            });
        }

        let edge_count = Self::count_internal_edges(graph);
        if edge_count < 1 {
            return Err(WholeApplicationRefusal::DisconnectedGraph {
                workload_id: self.id.to_string(),
                edge_count,
            });
        }

        Ok(())
    }

    /// Evaluate the graph on the host reference evaluator.
    ///
    /// This is the parity oracle. Nothing here contributes to a recorded
    /// latency, throughput, or memory figure.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when an external value has no supplied input bytes,
    /// when a node input is unresolved, or when the reference evaluator rejects
    /// a node program.
    pub fn evaluate_reference_outputs(
        &self,
        graph: &ProgramGraph,
        inputs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut value_map: BTreeMap<GraphValueId, Vec<u8>> = BTreeMap::new();

        for value in graph.values() {
            if value.producer.is_none() {
                let Some(bytes) = inputs.get(&value.name) else {
                    return Err(format!(
                        "workload `{}` supplies no input bytes for external value `{}`",
                        self.id, value.name
                    ));
                };
                value_map.insert(value.id, bytes.clone());
            }
        }

        for node in graph.nodes() {
            let mut node_inputs = Vec::new();
            for decl in node.program.buffers() {
                if !vyre_reference::is_reference_input(decl) {
                    continue;
                }
                let bytes = match node.inputs.iter().find(|port| port.buffer == decl.name()) {
                    Some(port) => {
                        let Some(bytes) = value_map.get(&port.value) else {
                            return Err(format!(
                                "workload `{}` node `{}` reads value {} before it is produced",
                                self.id, node.name, port.value.0
                            ));
                        };
                        bytes.clone()
                    }
                    None => {
                        let byte_len = decl
                            .static_byte_len()
                            .map_err(|err| {
                                format!(
                                    "workload `{}` node `{}` buffer `{}` has no static byte length: {err}",
                                    self.id,
                                    node.name,
                                    decl.name()
                                )
                            })?
                            .ok_or_else(|| {
                                format!(
                                    "workload `{}` node `{}` buffer `{}` declares a dynamic byte length",
                                    self.id,
                                    node.name,
                                    decl.name()
                                )
                            })?;
                        vec![0u8; byte_len]
                    }
                };
                node_inputs.push(Value::Bytes(Arc::from(bytes.into_boxed_slice())));
            }

            let node_outputs =
                vyre_reference::ReferenceRequest::standard(&node.program, &node_inputs)
                    .outputs()
                    .map_err(|err| {
                        format!("Reference eval failed on node `{}`: {:?}", node.name, err)
                    })?;

            for (out_value_id, port) in node.outputs.iter().zip(node.output_ports.iter()) {
                let Some(index) = vyre_reference::output_index(&node.program, &port.buffer) else {
                    return Err(format!(
                        "workload `{}` node `{}` declares output buffer `{}` that the oracle does not return",
                        self.id, node.name, port.buffer
                    ));
                };
                let Some(out_value) = node_outputs.get(index) else {
                    return Err(format!(
                        "reference eval of node `{}` produced {} outputs, buffer `{}` is at index {index}",
                        node.name,
                        node_outputs.len(),
                        port.buffer
                    ));
                };
                value_map.insert(*out_value_id, out_value.to_bytes());
            }
        }

        let mut outputs = BTreeMap::new();
        for value in graph.values() {
            if value.contract.lifetime == ValueLifetime::Output
                || (value.producer.is_some() && value.consumers.is_empty())
            {
                if let Some(bytes) = value_map.get(&value.id) {
                    outputs.insert(value.name.clone(), bytes.clone());
                }
            }
        }
        if outputs.is_empty() {
            return Err(format!(
                "workload `{}` graph declares no output value to compare",
                self.id
            ));
        }
        Ok(outputs)
    }

    /// Project one completion onto the named outputs the reference produced.
    fn device_outputs(
        &self,
        session: &ArtifactSession,
        completion: &Completion,
        names: &BTreeSet<&String>,
    ) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let mut outputs = BTreeMap::new();
        for name in names {
            let value = session.resource(name).map_err(|error| {
                format!("compiled artifact carries no resource named `{name}`: {error}")
            })?;
            let Some(bytes) = completion
                .outputs
                .get(&value)
                .or_else(|| completion.retained.get(&value))
            else {
                return Err(format!(
                    "device completion projected no bytes for output `{name}`"
                ));
            };
            outputs.insert((*name).clone(), bytes.clone());
        }
        Ok(outputs)
    }

    /// Compile, admit, and execute the whole application on one acquired device.
    ///
    /// Every recorded latency is the wall time of a submission the device
    /// completed. The reference evaluator runs once, for parity only.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic when the sample count is below
    /// [`MIN_MEASURED_SAMPLES`], when compilation, admission, binding, or a
    /// submission fails, when device outputs disagree with the reference, or
    /// when the resulting record fails validation.
    pub fn execute_and_measure(
        &self,
        device: &WholeApplicationDevice,
        measured_samples: usize,
    ) -> Result<WholeApplicationRecord, String> {
        if measured_samples < MIN_MEASURED_SAMPLES {
            return Err(format!(
                "workload `{}` requires at least {MIN_MEASURED_SAMPLES} measured samples, got {measured_samples}",
                self.id
            ));
        }

        let (graph, inputs) = (self.build_graph_and_inputs)()?;
        self.validate_topology(&graph)
            .map_err(|err| err.to_string())?;

        let input_hash = hash_named_bytes(&inputs);
        let input_digest = input_hash.to_hex().to_string();

        let request = CompileRequest::new(
            graph.clone(),
            ExternalFacts::new(Digest(*input_hash.as_bytes()), BTreeMap::new()),
            device.device_facts,
            SearchBudget::new(32, 1_000_000, 4, 0, 10_000_000),
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
        )
        .validate()
        .map_err(|err| format!("CompileRequest validation failed: {err:?}"))?;

        let compile_start = Instant::now();
        let artifact = compile(&request)
            .map_err(|err| format!("Compiler failed to produce artifact: {err:?}"))?;
        let compile_time_ns = elapsed_ns(compile_start);

        let envelope = attach_target(artifact.clone(), device.target_compiler.as_ref())
            .map_err(|err| format!("Target compilation failed for `{}`: {err:?}", self.id))?;

        let load_start = Instant::now();
        let session = ArtifactSession::from_envelope_with_materializer(
            device.registration,
            envelope,
            Arc::clone(&device.materializer),
        )
        .map_err(|err| format!("Artifact admission failed for `{}`: {err}", self.id))?;
        let load_time_ns = elapsed_ns(load_start);

        let artifact_digest = session
            .artifact()
            .map_err(|err| format!("Session reported no artifact identity: {err}"))?;
        let target_payload_digest = session
            .payload()
            .map_err(|err| format!("Session reported no payload identity: {err}"))?;
        let device_identity = session
            .device()
            .map_err(|err| format!("Session reported no device identity: {err}"))?;

        let workspace = session
            .allocate_workspace()
            .map_err(|err| format!("Artifact workspace allocation failed: {err}"))?;

        let mut dataset = TypedResourceDataset::new();
        for resource in artifact.resources() {
            if workspace.owns(resource.value) {
                continue;
            }
            let byte_count = usize::try_from(resource.byte_count).map_err(|_| {
                format!(
                    "resource `{}` declares {} bytes, which this host cannot address",
                    resource.name, resource.byte_count
                )
            })?;
            if byte_count == 0 {
                return Err(format!(
                    "resource `{}` declares no bytes, so nothing can be bound to it",
                    resource.name
                ));
            }
            let typed = match inputs.get(&resource.name) {
                Some(bytes) if bytes.len() == byte_count => {
                    TypedResource::memory(resource.value, bytes.clone())
                }
                Some(bytes) => {
                    return Err(format!(
                        "input `{}` supplies {} bytes, the artifact declares {byte_count}",
                        resource.name,
                        bytes.len()
                    ));
                }
                None => TypedResource::zeroed(resource.value, byte_count),
            };
            dataset
                .insert(typed.with_lifetime(resource.lifetime))
                .map_err(|err| format!("resource `{}` cannot be ingested: {err}", resource.name))?;
        }

        let bindings = session
            .ingest_with_workspace(&workspace, &dataset)
            .map_err(|err| format!("Resource ingestion failed for `{}`: {err}", self.id))?;
        let bound_resource_count = bindings.resources().len();
        let cold_start = Instant::now();
        session
            .submit_and_wait(bindings.clone())
            .map_err(|err| format!("Cold device submission failed for `{}`: {err}", self.id))?;
        let cold_latency_ns = elapsed_ns(cold_start);

        let mut wall_samples = Vec::with_capacity(measured_samples);
        let mut device_samples = Vec::with_capacity(measured_samples);
        let mut last_completion = None;
        for sample in 0..measured_samples {
            let start = Instant::now();
            let completion = session.submit_and_wait(bindings.clone()).map_err(|err| {
                format!(
                    "Device submission {sample} of {measured_samples} failed for `{}`: {err}",
                    self.id
                )
            })?;
            wall_samples.push(elapsed_ns(start));
            if let Some(device_ns) = completion.device_ns {
                device_samples.push(device_ns);
            }
            last_completion = Some(completion);
        }
        let completion = last_completion
            .ok_or_else(|| format!("workload `{}` completed no submission", self.id))?;

        wall_samples.sort_unstable();
        device_samples.sort_unstable();
        let p50_latency_ns = percentile(&wall_samples, 50)?;
        let p99_latency_ns = percentile(&wall_samples, 99)?;
        let device_p50_latency_ns = percentile(&device_samples, 50).ok();

        let reference_outputs = self.evaluate_reference_outputs(&graph, &inputs)?;
        let compared_names: BTreeSet<&String> = reference_outputs.keys().collect();
        let device_outputs = self.device_outputs(&session, &completion, &compared_names)?;
        let parity_result = WholeAppParityRecord::compare(&reference_outputs, &device_outputs)
            .map_err(|reason| {
                WholeApplicationRefusal::ParityMismatch {
                    workload_id: self.id.to_string(),
                    reason,
                }
                .to_string()
            })?;
        if !parity_result.is_exact_match {
            return Err(WholeApplicationRefusal::ParityMismatch {
                workload_id: self.id.to_string(),
                reason: format!(
                    "{} compared outputs differ, largest lane difference {}",
                    parity_result.compared_values, parity_result.max_ulp_distance
                ),
            }
            .to_string());
        }

        let peak_bytes = artifact.allocation().aggregate_peak_bytes;
        let resident_bytes = workspace.total_bytes();
        for (value, bound) in bindings.resources() {
            if workspace.owns(*value) {
                continue;
            }
            if let BoundResource::Resident(resource) = bound {
                session.free_resident(resource.clone()).map_err(|err| {
                    format!(
                        "Ingested resource release failed for value {}: {err}",
                        value.0
                    )
                })?;
            }
        }
        session
            .free_workspace(workspace)
            .map_err(|err| format!("Artifact workspace release failed: {err}"))?;
        let output_elements = artifact
            .resources()
            .iter()
            .filter(|resource| resource.lifetime == ResourceLifetime::Output)
            .try_fold(0u64, |total, resource| {
                total.checked_add(resource.element_count)
            })
            .ok_or_else(|| format!("workload `{}` output element count exceeds u64", self.id))?;

        let p50_seconds = (p50_latency_ns as f64) / 1e9;
        let throughput = WholeAppThroughputRecord {
            gflops: None,
            gb_per_sec: Some(
                (artifact.resource_envelope().total_bytes as f64) / (p50_latency_ns as f64),
            ),
            items_per_sec: Some((output_elements as f64) / p50_seconds),
        };

        let warmup_ratio = (cold_latency_ns as f64) / (p50_latency_ns as f64);
        let selected_schedule_id = format!(
            "schedule.{}",
            Digest(
                artifact
                    .selected_plan()
                    .schedule
                    .identity()
                    .map_err(|err| format!("Selected schedule has no identity: {err}"))?
            )
            .to_hex()
        );

        let catalog = whole_application_native_baselines();
        let (native_baseline_comparison, native_baseline_unmeasured) =
            match catalog.measured(self.pinned_native_baseline_id) {
                Ok(baseline) => (
                    Some(WholeAppNativeComparisonRecord::from_measured_baseline(
                        self.id,
                        baseline,
                        &self.default_conditions,
                        &input_digest,
                        p50_latency_ns,
                    )?),
                    None,
                ),
                Err(reason) => (
                    None,
                    Some(WholeAppNativeBaselineUnmeasured {
                        baseline_id: self.pinned_native_baseline_id.to_string(),
                        baseline_name: self.pinned_native_baseline_name.to_string(),
                        reason,
                    }),
                ),
            };

        let provenance =
            MeasurementProvenance::capture(device_identity.backend, &device_identity.device)?;

        let record = WholeApplicationRecord {
            schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
            workload_id: self.id.to_string(),
            workload_name: self.name.to_string(),
            domain: self.domain,
            graph_node_count: graph.nodes().len(),
            graph_edge_count: Self::count_internal_edges(&graph),
            compile_request_validated: true,
            production_route: WholeAppProductionRouteRecord {
                backend_id: device_identity.backend.to_string(),
                device_id: device_identity.device.clone(),
                device_generation: device_identity.generation,
                artifact_digest: artifact_digest.to_hex(),
                target_payload_digest: target_payload_digest.to_hex(),
                bound_resource_count,
                completed_submissions: measured_samples + 1,
            },
            parity_result,
            compile_time_ns,
            load_time_ns,
            measured_samples,
            p50_latency_ns,
            p99_latency_ns,
            device_p50_latency_ns,
            throughput,
            peak_bytes: Some(peak_bytes),
            resident_bytes: Some(resident_bytes),
            cold_state: WholeAppStateMetrics {
                latency_ns: cold_latency_ns,
                memory_bytes: None,
                is_cold_start: true,
                warmup_ratio: Some(warmup_ratio),
            },
            warm_state: WholeAppStateMetrics {
                latency_ns: p50_latency_ns,
                memory_bytes: None,
                is_cold_start: false,
                warmup_ratio: Some(warmup_ratio),
            },
            selected_schedule_id,
            input_digest,
            native_baseline_comparison,
            native_baseline_unmeasured,
            host_environment: provenance.host_environment,
            recorded_at_utc: provenance.recorded_at_utc,
        };

        record
            .validate_required_fields()
            .map_err(|err| format!("Generated record failed validation: {err}"))?;

        Ok(record)
    }
}
