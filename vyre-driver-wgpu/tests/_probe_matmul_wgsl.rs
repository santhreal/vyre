//! Test crate.

#[test]
fn dump_matmul_wgsl() {
    let p = vyre_libs::math::matmul("a", "b", "out", 4, 4, 4);
    let lowered = vyre_foundation::optimizer::optimize(p.clone())
        .expect("registered optimizer must converge");
    let wgsl = vyre_driver_wgpu::emit::lower(&lowered).expect("lower");
    println!("===WGSL===\n{wgsl}\n===END===");
}
