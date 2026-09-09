//! CLI binary driver for downstream model compiler prefill and decode execution.

use vyre_model_compiler::config::NamedModelConfig;
use vyre_model_compiler::manifest::CheckpointManifest;
use vyre_model_compiler::pipeline::ModelCompiler;
use vyre_model_compiler::workload::WorkloadEnvelope;

fn main() {
    println!("===== vyre-model-compiler: prefill and decode compilation driver =====");

    let models_to_run = [
        NamedModelConfig::DeepSeekV4Flash,
        NamedModelConfig::Llama3_1_8B,
        NamedModelConfig::Qwen2_5_7B,
        NamedModelConfig::Gemma2_9B,
        NamedModelConfig::Mistral7B,
        NamedModelConfig::ClipViTLarge14,
    ];

    for named in models_to_run {
        let mut config = named.config();
        // Use 4 representative layers for standalone demonstration compile
        config.num_layers = 4.min(config.num_layers);

        println!(
            "\n--- Compiling model: {} (family: {:?}) ---",
            config.name, config.family
        );
        let manifest = CheckpointManifest::from_config(&config);

        // 1. Prefill Phase
        let prefill_workload = WorkloadEnvelope::prefill(1, 32, config.max_seq_len);
        let prefill_compiled = ModelCompiler::compile_model(&config, &prefill_workload)
            .unwrap_or_else(|e| panic!("Prefill compilation failed for {}: {e}", config.name));

        println!(
            "  [Prefill] Artifact ID: {} | Entries: {} | Resources: {} bytes",
            prefill_compiled.digest(),
            prefill_compiled.entry_count(),
            prefill_compiled.required_resource_bytes()
        );

        let prefill_session = ModelCompiler::admit_model(&prefill_compiled, &manifest)
            .unwrap_or_else(|e| panic!("Prefill admission failed for {}: {e}", config.name));
        println!(
            "  [Prefill] Admitted resource buffers: {}",
            prefill_session.len()
        );

        // 2. Decode Phase (if language model with tokens)
        if config.vocab_size > 0 {
            let decode_workload = WorkloadEnvelope::decode(1, 128, config.max_seq_len);
            let decode_compiled = ModelCompiler::compile_model(&config, &decode_workload)
                .unwrap_or_else(|e| panic!("Decode compilation failed for {}: {e}", config.name));

            println!(
                "  [Decode]  Artifact ID: {} | Entries: {} | Resources: {} bytes",
                decode_compiled.digest(),
                decode_compiled.entry_count(),
                decode_compiled.required_resource_bytes()
            );

            let decode_session = ModelCompiler::admit_model(&decode_compiled, &manifest)
                .unwrap_or_else(|e| panic!("Decode admission failed for {}: {e}", config.name));
            println!(
                "  [Decode]  Admitted resource buffers: {}",
                decode_session.len()
            );
        }
    }

    println!("\n===== All prefill and decode targets compiled and admitted successfully =====");
}
