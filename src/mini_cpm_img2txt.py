import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class MiniPCMImg2Txt:
    MODEL_ID = "openbmb/MiniCPM-V-2"

    def __init__(self, question_list: list[str], temperature: float = 0.7):
        self.question_list = question_list
        self.question_list = self.__create_question_list()
        self.temperature = temperature

    def __create_question_list(self) -> list:
        ret = []
        for q in self.question_list:
            ret.append({"role": "user", "content": q})
        return ret

    def generate_captions(self, raw_image: Image):
        model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-2", trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        try:
            # For Nvidia GPUs support BF16 (like A100, H100, RTX3090
            model = model.to(device="cuda", dtype=torch.bfloat16)
        except Exception as e:
            # For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
            model = model.to(device="cuda", dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(
            MiniPCMImg2Txt.MODEL_ID, trust_remote_code=True
        )
        model.eval()

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        res, context, _ = model.chat(
            image=raw_image,
            msgs=self.question_list,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=self.temperature,
        )

        return res
