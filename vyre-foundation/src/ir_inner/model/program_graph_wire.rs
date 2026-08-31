//! Stable bounded wire codec for connected [`ProgramGraph`] compositions.

use super::op_signature::{BufferAccess, DataType};
use super::program::Program;
use super::program_graph::{
    GraphInput, GraphOutput, GraphValueId, ProgramGraph, ProgramGraphError, ShapeDim,
    ValueContract, ValueLifetime,
};

const MAGIC: &[u8; 4] = b"VGR0";
const VERSION: u16 = 2;
const MAX_GRAPH_WIRE_BYTES: usize = 256 * 1024 * 1024;
const MAX_GRAPH_ITEMS: usize = 1_000_000;
const MAX_PORTS_PER_NODE: usize = 1_000_000;
const MAX_NAME_BYTES: usize = 4_096;
const MAX_RANK: usize = 256;
const MAX_DTYPE_BYTES: usize = 65_536;

impl ProgramGraph {
    /// Encode this connected composition into canonical versioned bytes.
    ///
    /// Each node keeps its existing VIR0 [`Program`] encoding. Graph framing
    /// adds stable diagnostic names, typed identities, ports, and retained transitions.
    pub fn to_wire(&self) -> Result<Vec<u8>, ProgramGraphError> {
        self.encode_wire(false)
    }

    /// Encode complete graph semantics while excluding physical workgroup
    /// geometry from each executable node.
    pub(crate) fn logical_wire(&self) -> Result<Vec<u8>, ProgramGraphError> {
        self.encode_wire(true)
    }

    fn encode_wire(&self, normalize_workgroups: bool) -> Result<Vec<u8>, ProgramGraphError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());

        let external = self
            .values()
            .iter()
            .filter(|value| value.producer.is_none())
            .collect::<Vec<_>>();
        put_len(&mut bytes, external.len(), "external value count")?;
        for value in external {
            put_string(&mut bytes, &value.name)?;
            put_contract(&mut bytes, &value.contract)?;
        }

        put_len(&mut bytes, self.nodes().len(), "node count")?;
        for node in self.nodes() {
            if node.outputs.len() != node.output_ports.len() {
                return Err(wire_error(format!(
                    "node `{}` has {} value ids but {} output ports",
                    node.name,
                    node.outputs.len(),
                    node.output_ports.len()
                )));
            }
            put_string(&mut bytes, &node.name)?;
            let program = if normalize_workgroups && node.program.workgroup_size_is_schedule_only()
            {
                node.program.with_rewritten_workgroup_size_and_entry(
                    [1, 1, 1],
                    node.program.entry().to_vec(),
                )
            } else {
                node.program.clone()
            }
            .to_wire()
            .map_err(|error| wire_error(format!("node `{}` Program: {error}", node.name)))?;
            put_bytes(&mut bytes, &program, "Program bytes")?;
            put_len(&mut bytes, node.inputs.len(), "input port count")?;
            for input in &node.inputs {
                put_string(&mut bytes, &input.buffer)?;
                bytes.extend_from_slice(&input.value.0.to_le_bytes());
                put_contract(&mut bytes, &input.contract)?;
            }
            put_len(&mut bytes, node.output_ports.len(), "output port count")?;
            for (id, output) in node.outputs.iter().zip(&node.output_ports) {
                let value = self
                    .values()
                    .get(id.0 as usize)
                    .ok_or(ProgramGraphError::MissingValue(*id))?;
                if value.producer != Some(node.id)
                    || value.name != output.name
                    || value.contract != output.contract
                    || value.retained_successor_of != output.retained_successor_of
                {
                    return Err(wire_error(format!(
                        "node `{}` output `{}` disagrees with graph value {:?}",
                        node.name, output.name, id
                    )));
                }
                put_string(&mut bytes, &output.buffer)?;
                put_string(&mut bytes, &output.name)?;
                put_contract(&mut bytes, &output.contract)?;
                match output.retained_successor_of {
                    Some(prior) => {
                        bytes.push(1);
                        bytes.extend_from_slice(&prior.0.to_le_bytes());
                    }
                    None => bytes.push(0),
                }
            }
            if bytes.len() > MAX_GRAPH_WIRE_BYTES {
                return Err(wire_error(format!(
                    "graph encoding exceeds {MAX_GRAPH_WIRE_BYTES} bytes"
                )));
            }
        }
        Ok(bytes)
    }

    /// Decode and revalidate a canonical connected composition.
    pub fn from_wire(bytes: &[u8]) -> Result<Self, ProgramGraphError> {
        if bytes.len() > MAX_GRAPH_WIRE_BYTES {
            return Err(wire_error(format!(
                "graph wire input is {} bytes; maximum is {MAX_GRAPH_WIRE_BYTES}",
                bytes.len()
            )));
        }
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(wire_error("magic mismatch; expected VGR0"));
        }
        let version = reader.u16()?;
        if version != VERSION {
            return Err(wire_error(format!(
                "unsupported graph wire version {version}; expected {VERSION}"
            )));
        }

        let external_count = reader.bounded_len(MAX_GRAPH_ITEMS, "external value count")?;
        let mut external = Vec::with_capacity(external_count);
        for _ in 0..external_count {
            external.push((reader.string()?, reader.contract()?));
        }
        let mut graph = ProgramGraph::new();
        graph.add_external_values(external)?;

        let node_count = reader.bounded_len(MAX_GRAPH_ITEMS, "node count")?;
        for _ in 0..node_count {
            let name = reader.string()?;
            let program_bytes = reader.bytes(MAX_GRAPH_WIRE_BYTES, "Program bytes")?;
            let program = Program::from_wire(program_bytes)
                .map_err(|error| wire_error(format!("node `{name}` Program: {error}")))?;
            let input_count = reader.bounded_len(MAX_PORTS_PER_NODE, "input port count")?;
            let mut inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                inputs.push(GraphInput {
                    buffer: reader.string()?,
                    value: GraphValueId(reader.u32()?),
                    contract: reader.contract()?,
                });
            }
            let output_count = reader.bounded_len(MAX_PORTS_PER_NODE, "output port count")?;
            let mut outputs = Vec::with_capacity(output_count);
            for _ in 0..output_count {
                let buffer = reader.string()?;
                let output_name = reader.string()?;
                let contract = reader.contract()?;
                let retained_successor_of = match reader.u8()? {
                    0 => None,
                    1 => Some(GraphValueId(reader.u32()?)),
                    tag => {
                        return Err(wire_error(format!(
                            "retained successor tag is {tag}; expected 0 or 1"
                        )))
                    }
                };
                outputs.push(GraphOutput {
                    buffer,
                    name: output_name,
                    contract,
                    retained_successor_of,
                });
            }
            graph.add_node(name, program, inputs, outputs)?;
        }
        if reader.remaining() != 0 {
            return Err(wire_error(format!(
                "graph wire input has {} trailing bytes",
                reader.remaining()
            )));
        }
        Ok(graph)
    }
}

fn put_contract(bytes: &mut Vec<u8>, contract: &ValueContract) -> Result<(), ProgramGraphError> {
    let dtype = serde_json::to_vec(&contract.dtype)
        .map_err(|error| wire_error(format!("dtype encode failed: {error}")))?;
    if dtype.len() > MAX_DTYPE_BYTES {
        return Err(wire_error(format!(
            "dtype encoding is {} bytes; maximum is {MAX_DTYPE_BYTES}",
            dtype.len()
        )));
    }
    put_bytes(bytes, &dtype, "dtype bytes")?;
    put_len(bytes, contract.shape.len(), "tensor rank")?;
    for dimension in &contract.shape {
        match dimension {
            ShapeDim::Known(extent) => {
                bytes.push(0);
                bytes.extend_from_slice(&extent.to_le_bytes());
            }
            ShapeDim::Symbol(symbol) => {
                bytes.push(1);
                put_string(bytes, symbol)?;
            }
        }
    }
    bytes.push(access_tag(contract.access.clone())?);
    bytes.push(match contract.lifetime {
        ValueLifetime::Constant => 0,
        ValueLifetime::Invocation => 1,
        ValueLifetime::Retained => 2,
        ValueLifetime::Output => 3,
    });
    Ok(())
}

fn access_tag(access: BufferAccess) -> Result<u8, ProgramGraphError> {
    match access {
        BufferAccess::ReadOnly => Ok(0),
        BufferAccess::ReadWrite => Ok(1),
        BufferAccess::WriteOnly => Ok(2),
        BufferAccess::Uniform => Ok(3),
        _ => Err(wire_error(format!(
            "unsupported BufferAccess variant {access:?}"
        ))),
    }
}

fn put_len(bytes: &mut Vec<u8>, len: usize, label: &str) -> Result<(), ProgramGraphError> {
    let len = u32::try_from(len).map_err(|_| wire_error(format!("{label} exceeds u32")))?;
    bytes.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn put_bytes(bytes: &mut Vec<u8>, value: &[u8], label: &str) -> Result<(), ProgramGraphError> {
    put_len(bytes, value.len(), label)?;
    bytes.extend_from_slice(value);
    Ok(())
}

fn put_string(bytes: &mut Vec<u8>, value: &str) -> Result<(), ProgramGraphError> {
    if value.is_empty() || value.len() > MAX_NAME_BYTES {
        return Err(wire_error(format!(
            "graph name is {} bytes; expected 1..={MAX_NAME_BYTES}",
            value.len()
        )));
    }
    put_bytes(bytes, value.as_bytes(), "string length")
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ProgramGraphError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| wire_error("wire offset overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| wire_error("truncated graph wire input"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ProgramGraphError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ProgramGraphError> {
        let mut bytes = [0_u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, ProgramGraphError> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ProgramGraphError> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn bounded_len(&mut self, maximum: usize, label: &str) -> Result<usize, ProgramGraphError> {
        let len = self.u32()? as usize;
        if len > maximum {
            return Err(wire_error(format!(
                "{label} is {len}; maximum is {maximum}"
            )));
        }
        Ok(len)
    }

    fn bytes(&mut self, maximum: usize, label: &str) -> Result<&'a [u8], ProgramGraphError> {
        let len = self.bounded_len(maximum, label)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, ProgramGraphError> {
        let bytes = self.bytes(MAX_NAME_BYTES, "string length")?;
        if bytes.is_empty() {
            return Err(wire_error("graph name is empty"));
        }
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| wire_error(format!("graph name is not UTF-8: {error}")))
    }

    fn contract(&mut self) -> Result<ValueContract, ProgramGraphError> {
        let dtype_bytes = self.bytes(MAX_DTYPE_BYTES, "dtype bytes")?;
        let dtype: DataType = serde_json::from_slice(dtype_bytes)
            .map_err(|error| wire_error(format!("dtype decode failed: {error}")))?;
        let rank = self.bounded_len(MAX_RANK, "tensor rank")?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(match self.u8()? {
                0 => ShapeDim::Known(self.u64()?),
                1 => ShapeDim::Symbol(self.string()?),
                tag => {
                    return Err(wire_error(format!(
                        "shape dimension tag is {tag}; expected 0 or 1"
                    )))
                }
            });
        }
        let access = match self.u8()? {
            0 => BufferAccess::ReadOnly,
            1 => BufferAccess::ReadWrite,
            2 => BufferAccess::WriteOnly,
            3 => BufferAccess::Uniform,
            tag => return Err(wire_error(format!("unknown buffer access tag {tag}"))),
        };
        let lifetime = match self.u8()? {
            0 => ValueLifetime::Constant,
            1 => ValueLifetime::Invocation,
            2 => ValueLifetime::Retained,
            3 => ValueLifetime::Output,
            tag => return Err(wire_error(format!("unknown value lifetime tag {tag}"))),
        };
        Ok(ValueContract {
            dtype,
            shape,
            access,
            lifetime,
        })
    }
}

fn wire_error(message: impl Into<String>) -> ProgramGraphError {
    ProgramGraphError::Wire(message.into())
}
