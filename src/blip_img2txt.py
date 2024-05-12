from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipConfig,
    BlipTextConfig,
    BlipVisionConfig,
)


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

        self.text_config_kwargs = {
            "max_length": max_words,
            "min_length": min_words,
            "num_beams": search_beams,
            # "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "padding": "max_length",
        }

    def generate_caption(self, image: Image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        processor = BlipProcessor.from_pretrained(self.model_id)

        # https://huggingface.co/docs/transformers/model_doc/blip#transformers.BlipTextConfig
        config_text = BlipTextConfig.from_pretrained(self.model_id)
        config_text.update(self.text_config_kwargs)
        config_vision = BlipVisionConfig.from_pretrained(self.model_id)
        config = BlipConfig.from_text_vision_configs(config_text, config_vision)

        # Update model configuration
        model = BlipForConditionalGeneration.from_pretrained(
            self.model_id, config=config
        )
        model = model.to("cuda")

        inputs = processor(
            image,
            self.conditional_caption,
            return_tensors="pt",
        ).to("cuda")

        return processor.decode(model.generate(**inputs)[0], skip_special_tokens=True)
