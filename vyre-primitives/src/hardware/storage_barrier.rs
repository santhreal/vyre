//! Cat-C `storage_barrier`  -  storage-scope memory fence emitted after an
//! identity u32 store.

define_barrier_u32_hardware_intrinsic!(
    storage_barrier,
    "vyre-primitives::hardware::storage_barrier",
    &[10u32, 20, 30, 40],
    0xB200_0022,
    &[7]
);
