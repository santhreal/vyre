//! Parse C source and lower its `kernel` entry to backend-neutral Vyre IR.

use std::time::Instant;

use vyre_frontend_c::{lower_translation_unit, parse_source};

const HELPER_TEMPLATE: &str = "static unsigned int helper_$(NAME)(void) { return $(VALUE)u; }\n";

fn build_translation_unit(helper_count: usize) -> String {
    let mut source = String::with_capacity(helper_count.saturating_mul(64) + 64);
    for index in 0..helper_count {
        source.push_str(
            &HELPER_TEMPLATE
                .replace("$(NAME)", &format!("{index:04}"))
                .replace("$(VALUE)", &index.to_string()),
        );
    }
    source.push_str("unsigned int kernel(void) { return 6u * 7u; }\n");
    source
}

fn main() {
    let helper_count = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(128_usize);
    let source = build_translation_unit(helper_count);

    let start = Instant::now();
    let parsed = parse_source(&source).unwrap_or_else(|error| {
        eprintln!("C source ingestion failed: {error}");
        std::process::exit(2);
    });
    let program = lower_translation_unit(&parsed).unwrap_or_else(|error| {
        eprintln!("C typed-IR lowering failed: {error}");
        std::process::exit(2);
    });
    let elapsed = start.elapsed();

    println!("source bytes: {}", source.len());
    println!(
        "syntax nodes: {}",
        parsed.syntax_tree().root_node().descendant_count()
    );
    println!("IR buffers: {}", program.buffers.len());
    println!("IR statements: {}", program.entry.len());
    println!("source-to-IR: {elapsed:.2?}");
}
