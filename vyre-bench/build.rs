//! Export the optimization level cargo compiled this harness with.
//!
//! A report names the profile it was measured under, and the crate cannot see
//! that profile: `debug_assertions` answers a neighbouring question, and a
//! profile that turns assertions off without optimizing reads as release
//! through it. Cargo hands the build script the settled optimization level, so
//! the harness reads the level instead of inferring it.

fn main() {
    let opt_level = std::env::var("OPT_LEVEL").unwrap_or_else(|error| {
        eprintln!(
            "Fix: OPT_LEVEL is missing from the build-script environment ({error}); build vyre-bench with cargo."
        );
        std::process::exit(1);
    });
    println!("cargo:rustc-env=VYRE_BENCH_OPT_LEVEL={opt_level}");
}
