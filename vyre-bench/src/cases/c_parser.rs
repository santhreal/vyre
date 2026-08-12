use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;
use vyre_frontend_c::{lower_translation_unit, parse_source};

struct CSourceToIrCase {
    id: &'static str,
    name: &'static str,
    helper_count: usize,
}

struct CSourceToIrPrepared {
    source: String,
    program: vyre_foundation::ir::Program,
}

impl CSourceToIrCase {
    fn source(&self) -> String {
        let mut source = String::with_capacity(self.helper_count.saturating_mul(64) + 96);
        for index in 0..self.helper_count {
            source.push_str("static unsigned int helper_");
            source.push_str(&index.to_string());
            source.push_str("(void) { return ");
            source.push_str(&index.to_string());
            source.push_str("u; }\n");
        }
        source.push_str("unsigned int kernel(void) { return 6u * 7u; }\n");
        source
    }
}

impl BenchCase for CSourceToIrCase {
    fn id(&self) -> BenchId {
        BenchId(self.id.to_owned())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: self.name.to_owned(),
            description: "Backend-neutral C source ingestion and typed-IR construction".to_owned(),
            tags: vec![
                "frontend-c".to_owned(),
                "parser".to_owned(),
                "typed-ir".to_owned(),
                "source-ingestion".to_owned(),
            ],
            layer: BenchLayer::Foundation,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-frontend-c".to_owned(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        &[SuiteKind::Release, SuiteKind::Deep, SuiteKind::Honest]
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: false,
            needs_network: false,
            min_vram_bytes: None,
            min_input_bytes: None,
            feature_set: vec!["vyre-frontend-c".to_owned(), "source-to-ir".to_owned()],
        }
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let source = self.source();
        let parsed = parse_source(&source)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let program = lower_translation_unit(&parsed)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        Ok(Box::new(CSourceToIrPrepared { source, program }))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a vyre_foundation::ir::Program> {
        prepared
            .downcast_ref::<CSourceToIrPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        _ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<CSourceToIrPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "C source-to-IR prepared payload type mismatch".to_owned(),
                )
            })?;

        let start = std::time::Instant::now();
        let parsed = parse_source(&prepared.source)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let program = lower_translation_unit(&parsed)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
        let wall_ns = start.elapsed().as_nanos() as u64;
        let output = program
            .to_wire()
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(wall_ns),
                input_bytes: Some(prepared.source.len() as u64),
                output_bytes: Some(output.len() as u64),
                bytes_touched: Some(
                    (prepared.source.len() as u64).saturating_add(output.len() as u64),
                ),
                custom: vec![
                    MetricPoint {
                        name: "c_frontend_syntax_nodes".to_owned(),
                        value: syntax_node_count(parsed.syntax_tree().root_node()),
                    },
                    MetricPoint {
                        name: "c_frontend_ir_buffers".to_owned(),
                        value: program.buffers.len() as u64,
                    },
                ],
                ..Default::default()
            },
            baseline_metrics: None,
            outputs: vec![output],
            baseline_outputs: None,
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        let output = run.outputs.first().ok_or_else(|| {
            BenchError::CorrectnessViolation(
                "C source-to-IR benchmark produced no typed program bytes".to_owned(),
            )
        })?;
        if output.is_empty() {
            return Err(BenchError::CorrectnessViolation(
                "C source-to-IR benchmark produced empty typed program bytes".to_owned(),
            ));
        }
        Ok(Correctness::Certificate {
            digest: *blake3::hash(output).as_bytes(),
        })
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<CSourceToIrPrepared>()
            .map(|prepared| (prepared.source.len() as u64, 0))
            .unwrap_or((0, 0))
    }
}

fn syntax_node_count(root: tree_sitter::Node<'_>) -> u64 {
    let mut total = 1u64;
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        total = total.saturating_add(syntax_node_count(child));
    }
    total
}

static C_PARSER_PIPELINE: CSourceToIrCase = CSourceToIrCase {
    id: "frontend.c.parser.linux_driver_pipeline",
    name: "Vyre-C Source-to-IR Pipeline",
    helper_count: 1,
};

static C_PARSER_ONLY: CSourceToIrCase = CSourceToIrCase {
    id: "frontend.c.parser_only.linux_driver_pipeline",
    name: "Vyre-C Source Ingestion",
    helper_count: 8,
};

static C_PARSER_CORPUS: CSourceToIrCase = CSourceToIrCase {
    id: "frontend.c.parser_sema.linux_driver_corpus100",
    name: "Vyre-C Typed-IR Corpus",
    helper_count: 100,
};

inventory::submit! {
    &C_PARSER_PIPELINE as &'static dyn BenchCase
}

inventory::submit! {
    &C_PARSER_ONLY as &'static dyn BenchCase
}

inventory::submit! {
    &C_PARSER_CORPUS as &'static dyn BenchCase
}
