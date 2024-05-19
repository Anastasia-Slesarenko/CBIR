import torch
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from torch import nn, Tensor
from io import BytesIO
import PIL.Image as Image
from PIL.JpegImagePlugin import JpegImageFile


def extract_features_from_batch(
    batch: list[JpegImageFile],
    model_pth: str,
    device: str = "cpu",
) -> Tensor:
    """выдает эмбеддиинги батча. в случае инференса размер батча  равен 1"""
    # уточнить выходную размерность [batch_size, emb_size]
    model = torch.load(model_pth).to(device)
    transform = get_normalisation_resize_torch(im_size=224)
    batch = torch.cat([transform(image).unsqueeze(0) for image in batch], dim=0)
    model.eval()
    with torch.no_grad():
        out = model(batch.to(device)).to(device)
        out = out.cpu()
    return out


def extract_features_from_image(image: BytesIO, model_pth: str) -> Tensor:
    batch = Image.open(image)
    # batch = torch.tensor(image).squeeze(0)
    return extract_features_from_batch([batch], model_pth)


def extract_features_from_images(batch: Tensor, model_pth: str) -> Tensor:
    return extract_features_from_batch(batch, model_pth)
