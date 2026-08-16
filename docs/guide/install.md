# Install

```toml
[dependencies]
vyre = "0.7.2"
```

That gets the IR, the whole-program compiler, artifact admission and typed
submission. It gets no device.

## Add a backend

A concrete backend is a cargo feature on the facade:

```toml
[dependencies]
vyre = { version = "0.7.2", features = ["cuda"] }
```

```toml
[dependencies]
vyre = { version = "0.7.2", features = ["wgpu"] }
```

`cuda` pulls in `vyre-driver-cuda`. `wgpu` pulls in `vyre-driver-wgpu`.
The default feature set is empty, so a build that names no backend feature
links no concrete driver and every device acquisition fails closed.

`vyre-driver-spirv`, `vyre-driver-metal` and `vyre-driver-reference` are
published crates and are not facade features. A caller that wants one adds
it as a direct dependency.

## Why a feature is required

A backend registers itself at link time through the `inventory` crate.
Registrations live in the object file of the crate that declares them, and
a linker keeps that object only when a symbol inside it is referenced.
Naming a crate without calling into it is not a reference, so a binary that
merely mentions a driver reads a registry shorter than the workspace
declares. Enabling the feature is what makes the reference real.

`vyre-registry-link` is the one crate that names driver crates for linkage.
It reports the set it linked, so a narrower consumer states its set instead
of accepting a shorter registry in silence.

## Rust version

The published crates declare `rust-version = "1.85"`. The workspace pins
`1.86` in `rust-toolchain.toml`, which is the toolchain the gates run on.

## Where the crates are

Source: <https://github.com/santhreal/vyre>. API documentation:
<https://docs.rs/vyre>.
