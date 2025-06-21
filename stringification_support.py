import base64
import io

import torch


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Serializes a tensor to a base64 string."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _base64_to_tensor(b64_string: str) -> torch.Tensor:
    """Deserializes a base64 string to a tensor."""
    binary_data = base64.b64decode(b64_string.encode("utf-8"))
    buffer = io.BytesIO(binary_data)
    buffer.seek(0)
    try:
        return torch.load(buffer)  # Tries original device
    except RuntimeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
