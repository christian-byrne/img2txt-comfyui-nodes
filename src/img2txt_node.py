from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipConfig,
    BlipTextConfig,
    BlipVisionConfig,
)
from torchvision import transforms

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.img_tensor_utils import TensorImgUtils

# from termcolor import colored
from typing import Tuple


class Img2TxtNode:
    CATEGORY = "img2txt"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "conditional_image_captioning": (
                    "STRING",
                    {
                        "default": "a photograph of",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "min_words": ("INT", {"default": 36}),
                "max_words": ("INT", {"default": 128}),
                "search_beams": ("INT", {"default": 5}),
                "exclude_terms": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "skip_special_tokens": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Skip", "label_off": "Keep"},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        conditional_image_captioning: str,
        temperature: float,
        repetition_penalty: float,
        min_words: int,
        max_words: int,
        search_beams: int,
        exclude_terms: str,
        skip_special_tokens: bool,
        output_text: str = "",
        unique_id=None,
        extra_pnginfo=None,
    ) -> Tuple[str, ...]:
        """
        Args:
            input_image (torch.Tensor): [Batch_n, H, W, 3-channel] The input image
            conditional_image_captioning (str): Conditional captioning phrase
            temperature (float): 0.1 to 2.0 value to control the randomness of the output
            repetition_penalty (float): 0.1 to 2.0 value to control the repetition of the output
            min_words (int): Minimum number of tokens in the output
            max_words (int): Maximum number of tokens in the output
            search_beams (int): The number of beams in the beam search
            exclude_terms (str): Comma-separated terms to exclude from the output
            skip_special_tokens (bool): Whether to skip special tokens in the output like [CLS], [SEP], etc.
        """

        raw_image = transforms.ToPILImage()(
            TensorImgUtils.convert_to_type(input_image, "CHW")
        ).convert("RGB")

        if conditional_image_captioning == "":
            conditional_image_captioning = "a photograph of"

        general_caption = self.general_caption(
            raw_image,
            conditional_image_captioning,
            skip_special_tokens,
            min_words,
            max_words,
            temperature,
            repetition_penalty,
            search_beams,
        )

        out_string = self.exclude(exclude_terms, general_caption)

        return {"ui": {"text": out_string}, "result": (out_string,)}

    def exclude(self, exclude_terms: str, out_string: str) -> str:
        # https://huggingface.co/Salesforce/blip-image-captioning-large/discussions/20
        exclude_terms = "arafed," + exclude_terms
        exclude_terms = [
            term.strip().lower() for term in exclude_terms.split(",") if term != ""
        ]
        for term in exclude_terms:
            out_string = out_string.replace(term, "")

        return out_string

    def general_caption(
        self,
        rgb_input_image: Image.Image,
        conditional_caption: str,
        strip_special_tokens: bool,
        min_words: int,
        max_words: int,
        temperature: float,
        repetition_penalty: float,
        search_beams: int,
    ) -> str:
        model_path = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_path)

        # https://huggingface.co/docs/transformers/model_doc/blip#transformers.BlipTextConfig
        text_config_kwargs = {
            "max_length": max_words,
            "min_length": min_words,
            "num_beams": search_beams,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "padding": "max_length",
        }
        config_text = BlipTextConfig.from_pretrained(model_path)
        config_text.update(text_config_kwargs)
        config_vision = BlipVisionConfig.from_pretrained(model_path)
        config = BlipConfig.from_text_vision_configs(config_text, config_vision)

        # Update model configuration
        model = BlipForConditionalGeneration.from_pretrained(model_path, config=config)
        model = model.to("cuda")

        inputs = processor(
            rgb_input_image,
            conditional_caption,
            return_tensors="pt",
        ).to("cuda")

        out = model.generate(**inputs)
        out_string = processor.decode(out[0], skip_special_tokens=strip_special_tokens)

        return out_string
