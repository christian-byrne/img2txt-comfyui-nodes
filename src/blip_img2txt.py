from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipConfig,
    BlipTextConfig,
    BlipVisionConfig,
)

import torch
import model_management


class BLIPImg2Txt:
    def __init__(
        self,
        conditional_caption: str,
        min_words: int,
        max_words: int,
        temperature: float,
        repetition_penalty: float,
        search_beams: int,
        model_id: str = "Salesforce/blip-image-captioning-large",
    ):
        self.conditional_caption = conditional_caption
        self.model_id = model_id

        # Determine do_sample and num_beams
        if temperature > 1.1 or temperature < 0.90:
            do_sample = True
            num_beams = 1  # Sampling does not use beam search
        else:
            do_sample = False
            num_beams = (
                search_beams if search_beams > 1 else 1
            )  # Use beam search if num_beams > 1

        # Initialize text config kwargs
        self.text_config_kwargs = {
            "do_sample": do_sample,
            "max_length": max_words,
            "min_length": min_words,
            "repetition_penalty": repetition_penalty,
            "padding": "max_length",
        }
        if not do_sample:
            self.text_config_kwargs["temperature"] = temperature
            self.text_config_kwargs["num_beams"] = num_beams

    def generate_caption(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        processor = BlipProcessor.from_pretrained(self.model_id)

        # Update and apply configurations
        config_text = BlipTextConfig.from_pretrained(self.model_id)
        config_text.update(self.text_config_kwargs)
        config_vision = BlipVisionConfig.from_pretrained(self.model_id)
        config = BlipConfig.from_text_vision_configs(config_text, config_vision)

        model = BlipForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=torch.float16,
        ).to(model_management.get_torch_device())

        inputs = processor(
            image,
            self.conditional_caption,
            return_tensors="pt",
        ).to(model_management.get_torch_device(), torch.float16)

        with torch.no_grad():
            out = model.generate(**inputs)
            ret = processor.decode(out[0], skip_special_tokens=True)

        del model
        torch.cuda.empty_cache()

        return ret
