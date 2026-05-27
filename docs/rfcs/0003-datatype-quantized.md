# RFC 0003  -  DataType::Quantized

## Summary

Add a first-class quantization DataType:

```rust
DataType::Quantized {
    storage: Box<DataType>,          // underlying storage (I8 / I4 / U8 / FP4 / NF4)
    scale: QuantizationScale,        // PerTensor | PerChannel(u32) | PerGroup(u32)
    zero_point: QuantizationZeroPoint, // Absent | PerTensor | PerChannel(u32) | PerGroup(u32)
}
```

## Motivation

Modern LLM serving uses int8 weight-only quantization as standard;
int4 is common. Activation quantization is emerging. Vyre today
routes all weights + activations as u32/f32  -  inefficient for
every ML production workload.

Quantization also interacts with the other substrate RFCs: autodiff in
quantized domain needs straight-through estimators; Region
compositions need scale-aware rewrites; megakernel bytecode needs
tagged storage for quantized tensors.

## Design

New DataType variant: `Quantized { storage, scale, zero_point }`.

`ScaleKind`:
- `PerTensor`  -  one f32 scale per buffer
- `PerChannel(axis: u32)`  -  one scale per slice along axis
- `PerGroup(group_size: u32)`  -  one scale per `group_size` contiguous elements (GPTQ-style)

`ZeroPointKind`:
- `Absent`  -  symmetric quant, zero point = 0
- `PerTensor`  -  one zero point per buffer
- `PerChannel(axis: u32)`  -  one per slice
- `PerGroup(group_size: u32)`  -  one per contiguous quantization group

Quantized buffers carry two-to-three backing buffers:
- `<name>_storage`  -  the quantized values (I8 / I4 / etc.)
- `<name>_scale`  -  f32 scale factors
- `<name>_zero_point`  -  (optional) zero points

New BinOps: `QuantizedMatMul`, `QuantizedAdd`  -  backends lower these
to hardware tensor-core / MMA instructions when available, scalar
dequant-op-requant otherwise.

## Wire format

Tag `0x1F` is assigned to `Quantized`; `0x16..=0x18` are already shipped
sparse layout tags and must not be reused. Payload:
- storage DataType payload, restricted to I4/I8/I16/U8/U16/F8E4M3/F8E5M2/FP4/NF4
- 1 byte scale_kind discriminant + `u32` parameter for PerChannel/PerGroup
- 1 byte zero_point_kind discriminant + `u32` parameter for PerChannel/PerGroup

Dense memory-region shape payloads use the same logical fields with LEB128
integers so buffer declarations can round-trip without the generic recursive
type payload.

## Testing

- Round-trip: every supported `Quantized { ... }` storage/sidecar combination
  round-trips through wire format
- Parity: dequantize → op → quantize path matches the pure-float
  reference within a declared ULP budget
- Gap: every Category A nn op (`vyre-libs::nn::linear`, etc.) has
  a quantized variant registered

## Alternatives considered

- **Opaque extension only.** Rejected: every ML consumer needs
  quantization; making it an extension prevents cross-crate
  ecosystem composition.
- **Only f16/bf16 (no int quant).** Rejected: int4 weight-only is
  now standard for LLM serving; an ML-ready datatype surface needs int
  quantization.
- **Separate `vyre-quant` crate.** Considered; rejected because
  DataType is the cross-cutting surface.

## Open questions

- Int4 storage layout: nibble-packed vs byte-aligned?
- Quantization-aware training support belongs in the same source patch
  only if it has a conformance contract; otherwise it stays outside
  this datatype RFC.
