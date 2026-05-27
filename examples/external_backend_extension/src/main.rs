fn main() {
    let manifest = external_backend_extension::manifest();
    let wire = external_backend_extension::probe_wire()
        .unwrap_or_else(|error| panic!("Fix: external backend probe must encode: {error}"));
    println!(
        "{} {} targets {}; probe wire bytes={}",
        manifest.id,
        manifest.version,
        manifest.target,
        wire.len()
    );
}
