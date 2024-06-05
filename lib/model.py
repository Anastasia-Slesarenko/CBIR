from io import BytesIO
from typing import Union
import torch
from albumentations import Compose as aCompose
from oml.registry.transforms import get_transforms_for_pretrained
from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose as tCompose
from .settings import MODEL_NAME

TTransforms = Union[aCompose, tCompose]


def oml_transform(model_name: str) -> TTransforms:
    """
    Transforms images by resizing, converting to tensor
    and normalizing from pretrain model
    """
    transforms, _ = get_transforms_for_pretrained(model_name)
    return transforms


def extract_features_from_batch(
    batch: list[Image.Image],
    model: torch.nn.Module,
    device: str,
) -> Tensor:
    """
    Extracts embeddings from a batch of images.
    During inference, the batch size is 1.
    """
    transform = oml_transform(MODEL_NAME)
    batch = torch.cat(
        [transform(image).unsqueeze(0) for image in batch], dim=0
    )
    model.eval()
    with torch.no_grad():
        out = model(batch.to(device))
        out = out.cpu()
    return out


def extract_features_from_image(
    image: BytesIO,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Tensor:
    """
    Extracts the embedding of a single image.
    """
    batch = Image.open(image)
    return extract_features_from_batch(
        batch=[batch],
        model=model,
        device=device,
    )


def extract_features_from_images(
    batch: Tensor,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Tensor:
    """
    Extracts embeddings from a batch of images to filling the database.
    """
    return extract_features_from_batch(
        batch=batch,
        model=model,
        device=device,
    )
