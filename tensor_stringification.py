from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.invocation_api import (
    BaseInvocation,
    ConditioningFieldData,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    StringOutput,
    invocation,
)

from .stringification_support import (
    _base64_to_tensor,
    _deserialize_flux_conditioning_info,
    _serialize_flux_conditioning_info,
    _tensor_to_base64,
)


@invocation("latents_to_string", title="Latents to String", tags=["latents", "string"], version="1.0.0")
class LatentsToStringInvocation(BaseInvocation):
    """Convert latents to a base64-encoded string"""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        latents = context.tensors.load(self.latents.latents_name)
        return StringOutput(value=_tensor_to_base64(latents))


@invocation("string_to_latents", title="String to Latents", tags=["latents", "string"], version="1.0.0")
class StringToLatentsInvocation(BaseInvocation):
    """Convert a base64-encoded string back into latents"""

    value: str = InputField(description="Base64-encoded string representing latents.")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = _base64_to_tensor(self.value)
        latents_name = context.tensors.save(latents)
        return LatentsOutput.build(latents_name=latents_name, latents=latents, seed=None)


@invocation(
    "flux_conditioning_to_string", title="Flux Conditioning to String", tags=["conditioning", "string"], version="1.0.0"
)
class FluxConditioningToStringInvocation(BaseInvocation):
    """Convert FLUX conditioning to a base64-encoded string"""

    conditioning: FluxConditioningField = InputField(description="FLUX Conditioning input", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> StringOutput:
        cond_info = context.conditioning.load(self.conditioning.conditioning_name).conditionings[0]
        return StringOutput(value=_serialize_flux_conditioning_info(cond_info))


@invocation(
    "string_to_flux_conditioning", title="String to Flux Conditioning", tags=["conditioning", "string"], version="1.0.0"
)
class StringToFluxConditioningInvocation(BaseInvocation):
    """Convert a base64-encoded string back into FLUX conditioning"""

    value: str = InputField(description="Base64-encoded FLUXConditioningInfo object.")

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        cond_info = _deserialize_flux_conditioning_info(self.value)
        cond_data = ConditioningFieldData(conditionings=[cond_info])
        cond_name = context.conditioning.save(cond_data)
        return FluxConditioningOutput(conditioning=FluxConditioningField(conditioning_name=cond_name))
