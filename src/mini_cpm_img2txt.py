import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import model_management

class MiniPCMImg2Txt:
    def __init__(self, question_list: list[str], temperature: float = 0.7):
        self.model_id = "openbmb/MiniCPM-V-2"
        self.question_list = question_list
        self.question_list = self.__create_question_list()
        self.temperature = temperature

    def __create_question_list(self) -> list:
        ret = []
        for q in self.question_list:
            ret.append({"role": "user", "content": q})
        return ret

    def generate_captions(self, raw_image: Image.Image) -> str:
        device = model_management.get_torch_device()

        # For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
        # For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-2", trust_remote_code=True, torch_dtype=torch_dtype
        )
        model = model.to(device=device, dtype=torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model.eval()

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        with torch.no_grad():
            res, _, _ = model.chat(
                image=raw_image,
                msgs=self.question_list,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=self.temperature,
            )

        del model
        torch.cuda.empty_cache()

        return res
