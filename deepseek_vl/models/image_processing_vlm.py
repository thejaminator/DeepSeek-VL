from platform import processor
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from transformers import AutoImageProcessor, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array
from transformers.utils import logging

logger = logging.get_logger(__name__)

ImageType = Union[np.ndarray, torch.Tensor]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def expand2square(img: torch.Tensor, background_color):
    _, height, width = img.shape
    if width == height:
        return img
    elif width > height:
        result = torch.full((3, width, width), background_color, dtype=img.dtype)
        result[:, :, (width - height) // 2 : (width + height) // 2] = img
        return result
    else:
        result = torch.full((3, height, height), background_color, dtype=img.dtype)
        result[:, (height - width) // 2 : (height + width) // 2, :] = img
        return result


class VLMImageProcessorConfig(PretrainedConfig):
    model_type = "deepseek_vlm"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize
        self.default_shape = [3, self.image_size, self.image_size]


        self.background_color = tuple([1 for x in image_mean])

    def resize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): [3, H, W] in RGB

        Returns:
            x (torch.Tensor): [3, self.image_size, self.image_size]
        """

        _, height, width = img.shape
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {img.shape}, new size = {size}")
            raise ValueError("Invalid size!")

        x = torchvision.transforms.functional.resize(
            img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=True,
        )

        x = expand2square(x, self.background_color)

        return x
    
    def preprocess_one(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): Image tensor to be preprocessed, expected to be in [3, H, W] format
            Scale [0, 1]

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        assert isinstance(image, torch.Tensor), f"Input must be a PyTorch tensor, got {type(image)}"
        assert image.ndim == 3 and image.shape[0] == 3, "Input tensor should have shape [3, H, W]"

        # Resize and pad to [self.image_size, self.image_size]
        image = self.resize(image)

        # # Rescale from [0, 255] -> [0, 1]
        # image = image * self.rescale_factor

        # Normalize
        if self.do_normalize:
            image = (image - torch.tensor(self.image_mean, dtype=image.dtype).view(3, 1, 1)) / torch.tensor(self.image_std, dtype=image.dtype).view(3, 1, 1)

        return image
    
    # @property
    # def default_shape(self):
    #     return [3, self.image_size, self.image_size]
    
AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)
    
if __name__ == "__main__":
    from PIL import Image
    image_processor = VLMImageProcessor(
        image_size=1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True,
    )
    fake_image = Image.new("RGB", (1024, 1024), (1, 100, 200))
    # convert fake_image to torch.Tenso
    torch_image = torchvision.transforms.functional.to_tensor(fake_image)
    resized = image_processor.resize(torch_image)
    # print(resized)
    processed = image_processor.preprocess_one(torch_image)
    print(processed)
    print(image_processor.default_shape)
    