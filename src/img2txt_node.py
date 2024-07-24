"""
@author: christian-byrne
@title: Img2Txt auto captioning. Choose from models: BLIP, Llava, MiniCPM, MS-GIT. Use model combos and merge results. Specify questions to ask about images (medium, art style, background). Supports Chinese 🇨🇳 questions via MiniCPM.
@nickname: Image to Text - Auto Caption
"""

import torch
from torchvision import transforms

from .img_tensor_utils import TensorImgUtils
from .llava_img2txt import LlavaImg2Txt
from .blip_img2txt import BLIPImg2Txt
from .mini_cpm_img2txt import MiniPCMImg2Txt

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
                "use_blip_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Use BLIP (Requires 2Gb Disk)",
                        "label_off": "Don't use BLIP",
                    },
                ),
                "use_llava_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Use Llava (Requires 15Gb Disk)",
                        "label_off": "Don't use Llava",
                    },
                ),
                "use_mini_pcm_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Use MiniCPM (Requires 6Gb Disk)",
                        "label_off": "Don't use MiniCPM",
                    },
                ),
                "use_all_models": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Use all models and combine outputs (Total Size: 20+Gb)",
                        "label_off": "Use selected models only",
                    },
                ),
                "blip_caption_prefix": (
                    "STRING",
                    {
                        "default": "a photograph of",
                    },
                ),
                "prompt_questions": (
                    "STRING",
                    {
                        "default": "What is the subject of this image?\nWhat are the mediums used to make this?\nWhat are the artistic styles this is reminiscent of?\nWhich famous artists is this reminiscent of?\nHow sharp or detailed is this image?\nWhat is the environment and background of this image?\nWhat are the objects in this image?\nWhat is the composition of this image?\nWhat is the color palette in this image?\nWhat is the lighting in this image?",
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
                        "default": "watermark, text, writing",
                    },
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
        use_blip_model: bool,
        use_llava_model: bool,
        use_all_models: bool,
        use_mini_pcm_model: bool,
        blip_caption_prefix: str,
        prompt_questions: str,
        temperature: float,
        repetition_penalty: float,
        min_words: int,
        max_words: int,
        search_beams: int,
        exclude_terms: str,
        output_text: str = "",
        unique_id=None,
        extra_pnginfo=None,
    ) -> Tuple[str, ...]:
        raw_image = transforms.ToPILImage()(
            TensorImgUtils.convert_to_type(input_image, "CHW")
        ).convert("RGB")

        if blip_caption_prefix == "":
            blip_caption_prefix = "a photograph of"

        captions = []
        if use_all_models or use_blip_model:
            blip = BLIPImg2Txt(
                conditional_caption=blip_caption_prefix,
                min_words=min_words,
                max_words=max_words,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                search_beams=search_beams,
            )
            captions.append(blip.generate_caption(raw_image))

        if use_all_models or use_llava_model:
            llava_questions = prompt_questions.split("\n")
            llava_questions = [
                q
                for q in llava_questions
                if q != "" and q != " " and q != "\n" and q != "\n\n"
            ]
            if len(llava_questions) > 0:
                llava = LlavaImg2Txt(
                    question_list=llava_questions,
                    model_id="llava-hf/llava-1.5-7b-hf",
                    use_4bit_quantization=True,
                    use_low_cpu_mem=True,
                    use_flash2_attention=False,
                    max_tokens_per_chunk=300,
                )
                captions.append(llava.generate_caption(raw_image))

        if use_all_models or use_mini_pcm_model:
            mini_pcm = MiniPCMImg2Txt(
                question_list=prompt_questions.split("\n"),
                temperature=temperature,
            )
            captions.append(mini_pcm.generate_captions(raw_image))

        out_string = self.exclude(exclude_terms, self.merge_captions(captions))

        return {"ui": {"text": out_string}, "result": (out_string,)}

    def merge_captions(self, captions: list) -> str:
        """Merge captions from multiple models into one string.
        Necessary because we can expect the generated captions will generally
        be comma-separated fragments ordered by relevance - so combine
        fragments in an alternating order."""
        merged_caption = ""
        captions = [c.split(",") for c in captions]
        for i in range(max(len(c) for c in captions)):
            for j in range(len(captions)):
                if i < len(captions[j]) and captions[j][i].strip() != "":
                    merged_caption += captions[j][i].strip() + ", "
        return merged_caption

    def exclude(self, exclude_terms: str, out_string: str) -> str:
        # https://huggingface.co/Salesforce/blip-image-captioning-large/discussions/20
        exclude_terms = "arafed," + exclude_terms
        exclude_terms = [
            term.strip().lower() for term in exclude_terms.split(",") if term != ""
        ]
        for term in exclude_terms:
            out_string = out_string.replace(term, "")

        return out_string
