//! Cat-C `workgroup_barrier`  -  workgroup-scope memory fence emitted after an
//! identity u32 store.

define_barrier_u32_hardware_intrinsic!(
    workgroup_barrier,
    "vyre-primitives::hardware::workgroup_barrier",
    &[1u32, 2, 3, 4],
    &[
        0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00,
    ],
    0xB100_0011,
    &[42]
);
