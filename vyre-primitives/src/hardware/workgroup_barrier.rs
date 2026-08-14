//! Cat-C `workgroup_barrier`  -  workgroup-scope memory fence emitted after an
//! identity u32 store.

define_barrier_u32_hardware_intrinsic!(
    workgroup_barrier,
    "vyre-primitives::hardware::workgroup_barrier",
    &[1u32, 2, 3, 4],
    0xB100_0011,
    &[42]
);
