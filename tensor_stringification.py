from invokeai.invocation_api import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    StringOutput,
    invocation,
)

from .stringification_support import _base64_to_tensor, _tensor_to_base64


@invocation("latents_to_string", title="Latents to String", tags=["latents", "string"], version="1.0.0")
class LatentsToStringInvocation(BaseInvocation):
    """Convert latents to a base64-encoded string."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        latents = context.tensors.load(self.latents.latents_name)
        return StringOutput(value=_tensor_to_base64(latents))


@invocation("string_to_latents", title="String to Latents", tags=["latents", "string"], version="1.0.0")
class StringToLatentsInvocation(BaseInvocation):
    """Convert a base64-encoded string back into latents."""

    value: str = InputField(description="Base64-encoded string representing latents.")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = _base64_to_tensor(self.value)
        latents_name = context.tensors.save(latents)
        return LatentsOutput.build(latents_name=latents_name, latents=latents, seed=None)
