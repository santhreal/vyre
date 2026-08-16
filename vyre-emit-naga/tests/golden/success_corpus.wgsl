# Emitted-artifact byte-stability golden.
#
# One section per shared success-corpus case, in corpus order. Regenerate with
# the `bless_*` test in the file that reads this golden, then review the diff:
# a change here is a change in what the backend emits.
===== adv_deep_if_else
@group(0) @binding(0) 
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    var vyre_block_scope_1_: bool;
    var vyre_block_scope_10_: u32;
    var vyre_block_scope_11_: u32;
    var vyre_block_scope_12_: u32;
    var vyre_block_scope_13_: u32;
    var vyre_block_scope_20_: u32;
    var vyre_block_scope_21_: u32;

    if false {
        vyre_block_scope_1_ = true;
        let _e6 = vyre_block_scope_1_;
        if _e6 {
            vyre_block_scope_10_ = 7u;
            vyre_block_scope_11_ = 0u;
            let _e16 = vyre_block_scope_11_;
            let _e20 = vyre_block_scope_10_;
            out[_e16] = _e20;
        } else {
            vyre_block_scope_12_ = 13u;
            vyre_block_scope_13_ = 0u;
            let _e30 = vyre_block_scope_13_;
            let _e34 = vyre_block_scope_12_;
            out[_e30] = _e34;
        }
    } else {
        vyre_block_scope_20_ = 42u;
        vyre_block_scope_21_ = 1u;
        let _e44 = vyre_block_scope_21_;
        let _e48 = vyre_block_scope_20_;
        out[_e44] = _e48;
    }
}
===== adv_hostile_wg_1024
@group(0) @binding(0) 
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1024, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    out[_vyre_local_id.x] = (_vyre_local_id.x + 1u);
}
===== adv_multi_binding
@group(0) @binding(0) 
var<storage, read_write> u32_buf: array<u32>;
@group(0) @binding(1) 
var<storage, read_write> f32_buf: array<f32>;
@group(0) @binding(2) 
var<storage> const_u32_: array<u32>;

@compute @workgroup_size(128, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    let _e5 = const_u32_[0u];
    let _e8 = u32_buf[0u];
    let _e9 = (_e5 + _e8);
    u32_buf[0u] = _e9;
    let _e14 = f32_buf[0u];
    f32_buf[0u] = (_e14 + f32(_e9));
}
===== adv_shared_global_tile
@group(0) @binding(0) 
var<storage> global_in: array<u32>;
@group(0) @binding(1) 
var<storage, read_write> global_out: array<u32>;
var<workgroup> tile: array<u32, 256>;

@compute @workgroup_size(256, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    let _e4 = global_in[_vyre_local_id.x];
    tile[_vyre_local_id.x] = _e4;
    storageBarrier();
    workgroupBarrier();
    let _e9 = tile[_vyre_local_id.x];
    global_out[_vyre_local_id.x] = (_e9 + _e4);
}
===== adv_loop_barrier
@group(0) @binding(0) 
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(8, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    var i_end: u32;
    var i: u32;
    var vyre_block_scope_10_: u32;
    var vyre_block_scope_11_: u32;

    i_end = 4u;
    i = 0u;
    loop {
        let _e5 = i;
        let _e7 = i_end;
        if (_e5 >= _e7) {
            break;
        }
        storageBarrier();
        vyre_block_scope_10_ = _vyre_local_id.x;
        vyre_block_scope_11_ = 7u;
        let _e19 = vyre_block_scope_10_;
        let _e23 = vyre_block_scope_11_;
        out[_e19] = _e23;
        continuing {
            let _e25 = i;
            i = (_e25 + 1u);
            let _e29 = i;
            let _e31 = i_end;
            break if (_e29 >= _e31);
        }
    }
}
===== adv_atomic_counter
@group(0) @binding(0) 
var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    atomicAdd((&counter[0u]), 1u);
}
===== adv_dead_identity
@group(0) @binding(0) 
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    out[0u] = 99u;
}
===== adv_vec_load_fusion
@group(0) @binding(0) 
var<storage> input: array<u32>;
@group(0) @binding(1) 
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    let _e4 = input[0u];
    let _e8 = input[1u];
    let _e12 = input[2u];
    let _e16 = input[3u];
    output[0u] = (((_e4 + _e8) + _e12) + _e16);
}
===== adv_signed_buffer_arith
@group(0) @binding(0) 
var<storage> src: array<i32>;
@group(0) @binding(1) 
var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) _vyre_global_id: vec3<u32>, @builtin(workgroup_id) _vyre_workgroup_id: vec3<u32>, @builtin(local_invocation_id) _vyre_local_id: vec3<u32>) {
    let _e3 = src[0u];
    out[0u] = u32((((_e3 & i32(255u)) >> 3u) + i32(1u)));
}
