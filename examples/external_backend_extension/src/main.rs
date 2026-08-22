fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = external_backend_extension::dispatch_probe(&[1, 2, 3, 4])?;
    let backend = vyre_driver::acquire(external_backend_extension::BACKEND_ID)?;
    println!(
        "acquired {} {} out of tree; probe output {words:?}",
        backend.id(),
        backend.version()
    );
    Ok(())
}
