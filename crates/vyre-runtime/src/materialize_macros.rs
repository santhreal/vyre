//! Macros that expand to the structural methods every backend's materialized
//! instance and materializer write the same way.
//!
//! Each one expands to associated items inside the backend's own impl block,
//! so a backend states only the methods that differ.

/// Answer the three [`crate::ArtifactInstance`] identity methods from an
/// [`InstanceCore`](crate::materialize::InstanceCore) field named `core`.
///
/// Every backend forwards these three the same way, because the identity of a
/// materialized instance is exactly what its core recorded at materialization.
/// Only `submit` is a per-backend decision, so this expands to associated items
/// inside the backend's own `impl ArtifactInstance` block rather than to a whole
/// impl:
///
/// ```ignore
/// impl ArtifactInstance for TargetArtifactInstance {
///     vyre_runtime::artifact_instance_identity!();
///
///     fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
///         // the one method that differs
///     }
/// }
/// ```
#[macro_export]
macro_rules! artifact_instance_identity {
    () => {
        fn artifact(&self) -> ::vyre_compiler::Digest {
            self.core.artifact
        }

        fn payload(&self) -> ::vyre_compiler::Digest {
            self.core.payload
        }

        fn device(&self) -> &$crate::DeviceIdentity {
            &self.core.device
        }
    };
}

/// Answer [`crate::ArtifactInstance::emitted_resources`] with one unreported
/// record per module, and
/// [`crate::ArtifactInstance::resident_device_bytes`] with no figure.
///
/// A backend whose API exposes no register, spill or static shared figure for a
/// loaded entry point answers this way. The record count still has to match the
/// module count, because the compiler pairs the records with payload entries by
/// position, and zero in every field is what leaves the analytic estimate in
/// force for that entry.
///
/// A backend with no memory query reports no resident figure, which leaves the
/// planned allocation unreconciled rather than contradicted. A backend that can
/// answer states its own body instead of expanding this macro.
#[macro_export]
macro_rules! artifact_instance_unreported_resources {
    () => {
        fn emitted_resources(
            &self,
        ) -> ::std::result::Result<
            ::std::vec::Vec<::vyre_compiler::EmittedResources>,
            $crate::BackendError,
        > {
            ::std::result::Result::Ok(::std::vec![
                ::vyre_compiler::EmittedResources::default();
                $crate::materialize::MaterializedInstance::modules(self).len()
            ])
        }

        fn resident_device_bytes(
            &self,
        ) -> ::std::result::Result<::std::option::Option<u64>, $crate::BackendError> {
            ::std::result::Result::Ok(::std::option::Option::None)
        }
    };
}

/// Answer [`crate::ArtifactMaterializer::device`] from a
/// [`crate::materialize::MaterializerDevice`]
/// field named `descriptor`, and optionally forward the four resident-resource
/// methods to a [`crate::VyreBackend`] field.
///
/// The device accessor is the same line in every backend, because the device a
/// materializer reports is exactly the descriptor it was acquired with. The
/// resident four are the same four bodies wherever the backend has a resident
/// path at all: the materializer owns no allocator of its own, so each one
/// names the backend field and forwards unchanged. A backend without a resident
/// path passes no field and keeps the trait's refusals.
///
/// ```ignore
/// impl ArtifactMaterializer for TargetMaterializer {
///     vyre_runtime::materializer_passthrough!(resident);
///
///     fn materialize(&self, artifact: &Artifact, payload: &TargetPayload)
///         -> Result<Box<dyn ArtifactInstance>, BackendError> { /* per backend */ }
/// }
/// ```
#[macro_export]
macro_rules! materializer_passthrough {
    () => {
        fn device(&self) -> &dyn $crate::Device {
            &self.descriptor
        }
    };
    ($backend:ident) => {
        $crate::materializer_passthrough!();

        fn allocate_resident(
            &self,
            byte_len: usize,
        ) -> ::std::result::Result<$crate::Resource, $crate::BackendError> {
            $crate::VyreBackend::allocate_resident(&self.$backend, byte_len)
        }

        fn upload_resident(
            &self,
            resource: &$crate::Resource,
            bytes: &[u8],
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::upload_resident(&self.$backend, resource, bytes)
        }

        fn upload_resident_at(
            &self,
            resource: &$crate::Resource,
            offset_bytes: usize,
            bytes: &[u8],
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::upload_resident_at(&self.$backend, resource, offset_bytes, bytes)
        }

        fn free_resident(
            &self,
            resource: $crate::Resource,
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::free_resident(&self.$backend, resource)
        }
    };
}

/// Answer [`ResidentInstance::launch_resident`](crate::materialize::ResidentInstance::launch_resident) from a module field named
/// `pipeline` that implements [`crate::CompiledPipeline`].
///
/// A resident launch through a compiled pipeline is one call taking the ordered
/// handles and the config the shared path already built, so the backends whose
/// module holds a pipeline wrote the same body. A backend whose module holds
/// something else writes the method by hand.
///
/// ```ignore
/// impl ResidentInstance for TargetArtifactInstance {
///     vyre_runtime::resident_pipeline_launch!();
///
///     fn resident_module_label(&self) -> &'static str { /* per backend */ }
/// }
/// ```
#[macro_export]
macro_rules! resident_pipeline_launch {
    () => {
        fn launch_resident(
            &self,
            module: &Self::Module,
            ordered: &[$crate::Resource],
            config: &$crate::DispatchConfig,
        ) -> ::std::result::Result<$crate::TimedDispatchResult, $crate::BackendError> {
            $crate::CompiledPipeline::dispatch_persistent_handles_timed(
                module.pipeline.as_ref(),
                ordered,
                config,
            )
        }

        fn launch_resident_batch(
            &self,
            module: &Self::Module,
            batches: &[&[$crate::Resource]],
            config: &$crate::DispatchConfig,
        ) -> ::std::result::Result<
            ::std::vec::Vec<(
                $crate::OutputBuffers,
                ::std::option::Option<u64>,
            )>,
            $crate::BackendError,
        > {
            let mut outputs = ::std::vec::Vec::new();
            let mut device_ns_by_item = ::std::vec::Vec::new();
            $crate::CompiledPipeline::dispatch_persistent_handles_batched_timed(
                module.pipeline.as_ref(),
                batches,
                config,
                &mut outputs,
                &mut device_ns_by_item,
            )?;
            if outputs.len() != batches.len() || device_ns_by_item.len() != batches.len() {
                return ::std::result::Result::Err($crate::BackendError::InvalidProgram {
                    fix: ::std::format!(
                        "Fix: resident batch of {} item(s) reported {} output set(s) and {} timing(s). The backend's batched dispatch must report one of each per item.",
                        batches.len(),
                        outputs.len(),
                        device_ns_by_item.len(),
                    ),
                });
            }
            ::std::result::Result::Ok(
                outputs.into_iter().zip(device_ns_by_item).collect(),
            )
        }
    };
}

/// The `core` accessor and the staged-input `gather` a materialized instance
/// whose module carries `input_slots` implements.
///
/// Both are structural. `core` hands back the field the trait requires, and
/// `gather` resolves the slots through the one staging primitive. Written out
/// per backend they are the same text in each driver, and a backend that
/// diverged there would stage a fused artifact's inter-module values by a
/// second rule: one driver did, and seven operations read the Program's
/// host-input order instead of the target binding order.
///
/// ```ignore
/// impl MaterializedInstance for TargetArtifactInstance {
///     type Module = TargetExecutableModule;
///
///     vyre_runtime::staged_input_gather!();
///
///     fn modules(&self) -> &[Self::Module] { /* per backend */ }
/// }
/// ```
#[macro_export]
macro_rules! staged_input_gather {
    () => {
        fn core(&self) -> &$crate::materialize::InstanceCore {
            &self.core
        }

        fn gather<'a>(
            &'a self,
            module_index: usize,
            module: &'a Self::Module,
            _plan: &$crate::BindingPlan,
            state: &'a ::std::collections::BTreeMap<
                ::vyre_compiler::ArtifactValueId,
                ::std::vec::Vec<u8>,
            >,
        ) -> ::std::result::Result<::std::vec::Vec<&'a [u8]>, $crate::BackendError> {
            $crate::materialize::gather_artifact_inputs(
                &self.core,
                module_index,
                &module.input_slots,
                state,
            )
        }
    };
}
