import torch
from typing import Tuple


class TensorImgUtils:
    @staticmethod
    def from_to(from_type: list[str], to_type: list[str]):
        """Return a function that converts a tensor from one type to another. Args can be lists of strings or just strings (e.g., ["C", "H", "W"] or just "CHW")."""
        if isinstance(from_type, list):
            from_type = "".join(from_type)
        if isinstance(to_type, list):
            to_type = "".join(to_type)

        permute_arg = [from_type.index(c) for c in to_type]

        def convert(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(permute_arg)

        return convert

    @staticmethod
    def convert_to_type(tensor: torch.Tensor, to_type: str) -> torch.Tensor:
        """Convert a tensor to a specific type."""
        from_type = TensorImgUtils.identify_type(tensor)[0]
        if from_type == list(to_type):
            return tensor

        if len(from_type) == 4 and len(to_type) == 3:
            # If converting from a batched tensor to a non-batched tensor, squeeze the batch dimension
            tensor = tensor.squeeze(0)
            from_type = from_type[1:]
        if len(from_type) == 3 and len(to_type) == 4:
            # If converting from a non-batched tensor to a batched tensor, unsqueeze the batch dimension
            tensor = tensor.unsqueeze(0)
            from_type = ["B"] + from_type

        return TensorImgUtils.from_to(from_type, list(to_type))(tensor)

    @staticmethod
    def identify_type(tensor: torch.Tensor) -> Tuple[list[str], str]:
        """Identify the type of image tensor. Doesn't currently check for BHW. Returns one of the following:"""
        dim_n = tensor.dim()
        if dim_n == 2:
            return (["H", "W"], "HW")
        elif dim_n == 3:  # HWA, AHW, HWC, or CHW
            if tensor.size(2) == 3:
                return (["H", "W", "C"], "HWRGB")
            elif tensor.size(2) == 4:
                return (["H", "W", "C"], "HWRGBA")
            elif tensor.size(0) == 3:
                return (["C", "H", "W"], "RGBHW")
            elif tensor.size(0) == 4:
                return (["C", "H", "W"], "RGBAHW")
            elif tensor.size(2) == 1:
                return (["H", "W", "C"], "HWA")
            elif tensor.size(0) == 1:
                return (["C", "H", "W"], "AHW")
        elif dim_n == 4:  # BHWC or BCHW
            if tensor.size(3) >= 3:  # BHWRGB or BHWRGBA
                if tensor.size(3) == 3:
                    return (["B", "H", "W", "C"], "BHWRGB")
                elif tensor.size(3) == 4:
                    return (["B", "H", "W", "C"], "BHWRGBA")

            elif tensor.size(1) >= 3:
                if tensor.size(1) == 3:
                    return (["B", "C", "H", "W"], "BRGBHW")
                elif tensor.size(1) == 4:
                    return (["B", "C", "H", "W"], "BRGBAHW")

        else:
            raise ValueError(
                f"{dim_n} dimensions is not a valid number of dimensions for an image tensor."
            )

        raise ValueError(
            f"Could not determine shape of Tensor with {dim_n} dimensions and {tensor.shape} shape."
        )

    @staticmethod
    def test_squeeze_batch(tensor: torch.Tensor, strict=False) -> torch.Tensor:
        # Check if the tensor has a batch dimension (size 4)
        if tensor.dim() == 4:
            if tensor.size(0) == 1 or not strict:
                # If it has a batch dimension with size 1, remove it. It represents a single image.
                return tensor.squeeze(0)
            else:
                raise ValueError(
                    f"This is not a single image. It's a batch of {tensor.size(0)} images."
                )
        else:
            # Otherwise, it doesn't have a batch dimension, so just return the tensor as is.
            return tensor

    @staticmethod
    def test_unsqueeze_batch(tensor: torch.Tensor) -> torch.Tensor:
        # Check if the tensor has a batch dimension (size 4)
        if tensor.dim() == 3:
            # If it doesn't have a batch dimension, add one. It represents a single image.
            return tensor.unsqueeze(0)
        else:
            # Otherwise, it already has a batch dimension, so just return the tensor as is.
            return tensor

    @staticmethod
    def most_pixels(img_tensors: list[torch.Tensor]) -> torch.Tensor:
        sizes = [
            TensorImgUtils.height_width(img)[0] * TensorImgUtils.height_width(img)[1]
            for img in img_tensors
        ]
        return img_tensors[sizes.index(max(sizes))]

    @staticmethod
    def height_width(image: torch.Tensor) -> Tuple[int, int]:
        """Like torchvision.transforms methods, this method assumes Tensor to
        have [..., H, W] shape, where ... means an arbitrary number of leading
        dimensions
        """
        return image.shape[-2:]

    @staticmethod
    def smaller_axis(image: torch.Tensor) -> int:
        h, w = TensorImgUtils.height_width(image)
        return 2 if h < w else 3

    @staticmethod
    def larger_axis(image: torch.Tensor) -> int:
        h, w = TensorImgUtils.height_width(image)
        return 2 if h > w else 3
