from albumentations import Compose as aCompose
from io import BytesIO
import torch
from torchvision.transforms import Normalize, ToTensor, Resize
from torchvision.transforms import Compose as tCompose
from torch import Tensor
from PIL import Image
from .settings import IMAGE_SIZE, MEAN, STD, TNormParam, MODEL_NAME
from oml.registry.transforms import get_transforms_for_pretrained
from typing import Union


TTransforms = Union[aCompose, tCompose]


def get_normalisation_resize_torch(
    im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD
) -> tCompose:
    """
    Transforms images by resizing, converting to tensor, and normalizing.
    """
    return tCompose(
        [
            Resize(size=(im_size, im_size), antialias=True),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def oml_transform(model_name: str) -> TTransforms:
    """
    Transforms images by resizing, converting to tensor, and normalizing from pretrain model 
    """
    transforms, _ = get_transforms_for_pretrained(model_name)
    return transforms


MODEL_TRANSFORM = {
    "vit": get_normalisation_resize_torch(im_size=IMAGE_SIZE),
    "vitl14_336px_unicom": oml_transform("vitl14_336px_unicom"),
}


def extract_features_from_batch(
    batch: list[Image.Image],
    model: torch.nn.Module,
    device: str,
) -> Tensor:
    """
    Extracts embeddings from a batch of images.
    During inference, the batch size is 1.
    """
    transform = MODEL_TRANSFORM[MODEL_NAME]
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
