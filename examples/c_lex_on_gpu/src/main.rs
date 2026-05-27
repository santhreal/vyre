//! Real C parsed end-to-end on the GPU.
//!
//! Runs C source through `vyre-frontend-c`'s GPU lex + AST pipeline
//! (lex -> keyword -> brackets -> structure -> AST). There is no CPU
//! fallback for the vyre side. If the GPU probe fails the demo exits
//! with an actionable error instead of silently falling back.
//!
//! Usage:
//!     c_lex_on_gpu          # default: 1 copy (smallest TU  -  always succeeds)
//!     c_lex_on_gpu 3        # 3 copies (~500 B)  -  also succeeds
//!     c_lex_on_gpu 5        # 5 copies (~1 KiB)  -  exercises multi-function TU parsing
//!
//! The demo proves the end-to-end GPU path runs without a host-reference
//! escape path and reports cold/warm timings so throughput regressions are
//! visible during local release checks.

use std::time::Instant;

use vyre_frontend_c::api::SyntaxParseSummary;

// Force the GPU backend crates to link. Without these `use`s the
// linker would strip their `inventory::submit!` registrations and
// `acquire_preferred_dispatch_backend` would see an empty registry.
#[allow(unused_imports)]
use vyre_driver_cuda as _;
#[allow(unused_imports)]
use vyre_driver_wgpu as _;

const FUNCTION_TEMPLATE: &str = "\
int compute_$(NAME)(int a, int b) {
    int c = a + b;
    int d = a * b;
    int e = c - d;
    if (e > 0) {
        return e + 1;
    } else {
        return e - 1;
    }
}

";

fn build_translation_unit(copies: usize) -> String {
    let mut tu = String::with_capacity(copies * FUNCTION_TEMPLATE.len() + 128);
    tu.push_str("// vyre c_lex_on_gpu demo translation unit\n\n");
    for i in 0..copies {
        tu.push_str(&FUNCTION_TEMPLATE.replace("$(NAME)", &format!("fn_{i:04}")));
    }
    tu.push_str("int main(void) {\n    return compute_fn_0000(1, 2);\n}\n");
    tu
}

fn main() {
    let copies = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize);
    let source = build_translation_unit(copies);
    let bytes = source.as_bytes();
    let lines = source.lines().count();
    let kib = bytes.len() as f64 / 1024.0;

    println!("vyre  -  real C, parsed end-to-end on GPU");
    println!("========================================");
    println!(
        "input: {copies} copies of a 9-line C function = {kib:.2} KiB / {lines} lines"
    );
    println!();

    let cold_start = Instant::now();
    let cold_summary = match vyre_frontend_c::api::parse_syntax_bytes(bytes) {
        Ok(s) => s,
        Err(error) => {
            eprintln!("vyre GPU parse failed: {error}");
            std::process::exit(2);
        }
    };
    let cold = cold_start.elapsed();

    let warm_iters = 32u32;
    let warm_start = Instant::now();
    let mut last: SyntaxParseSummary = cold_summary;
    for _ in 0..warm_iters {
        last = vyre_frontend_c::api::parse_syntax_bytes(bytes)
            .expect("Fix: warm parse must succeed once cold parse did");
    }
    let warm_total = warm_start.elapsed();
    let warm_per_call = warm_total / warm_iters;

    let mb = bytes.len() as f64 / (1024.0 * 1024.0);
    let warm_mbps = mb / warm_per_call.as_secs_f64();
    let cold_mbps = mb / cold.as_secs_f64();

    println!("backend:          {}", last.backend_id);
    println!("tokens:           {}", last.token_count);
    println!("AST nodes:        {}", last.ast_node_count);
    println!(
        "AST coverage:     {} / {} tokens",
        last.ast_covered_tokens, last.token_count
    );
    println!("vyre GPU parse");
    println!(
        "  cold (1 call):  {:>9.2?}   ({:>6.1} MiB/s)   cache miss + GPU warmup + kernel compile",
        cold, cold_mbps
    );
    println!(
        "  warm (avg/{warm_iters}):  {:>9.2?}   ({:>6.1} MiB/s)   pipeline cache hit, kernels resident",
        warm_per_call, warm_mbps
    );
    println!();
    println!("Vyre lexed and built an AST entirely on GPU.");
    println!("There is no host-reference escape path. Backend chosen by runtime probe: {}.", last.backend_id);
}
