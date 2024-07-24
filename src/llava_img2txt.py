from PIL import Image
import torch
import model_management
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


class LlavaImg2Txt:
    """
    A class to generate text captions for images using the Llava model.

    Args:
        question_list (list[str]): A list of questions to ask the model about the image.
        model_id (str): The model's name in the Hugging Face model hub.
        use_4bit_quantization (bool): Whether to use 4-bit quantization to reduce memory usage. 4-bit quantization reduces the precision of model parameters, potentially affecting the quality of generated outputs. Use if VRAM is limited. Default is True.
        use_low_cpu_mem (bool): In low_cpu_mem_usage mode, the model is initialized with optimizations aimed at reducing CPU memory consumption. This can be beneficial when working with large models or limited computational resources. Default is True.
        use_flash2_attention (bool): Whether to use Flash-Attention 2. Flash-Attention 2 focuses on optimizing attention mechanisms, which are crucial for the model's performance during generation. Use if computational resources are abundant. Default is False.
        max_tokens_per_chunk (int): The maximum number of tokens to generate per prompt chunk. Default is 300.
    """

    def __init__(
        self,
        question_list,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        use_4bit_quantization: bool = True,
        use_low_cpu_mem: bool = True,
        use_flash2_attention: bool = False,
        max_tokens_per_chunk: int = 300,
    ):
        self.question_list = question_list
        self.model_id = model_id
        self.use_4bit = use_4bit_quantization
        self.use_flash2 = use_flash2_attention
        self.use_low_cpu_mem = use_low_cpu_mem
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def generate_caption(
        self,
        raw_image: Image.Image,
    ) -> str:
        """
        Generate a caption for an image using the Llava model.

        Args:
            raw_image (Image): Image to generate caption for
        """
        # Convert Image to RGB first
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="fp4"
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=self.use_low_cpu_mem,
            use_flash_attention_2=self.use_flash2,
            quantization_config=quant_config,
        )

        # model.to() is not supported for 4-bit or 8-bit bitsandbytes models. With 4-bit quantization, use the model as it is, since the model will already be set to the correct devices and casted to the correct `dtype`.
        if torch.cuda.is_available() and not self.use_4bit:
            model = model.to(model_management.get_torch_device(), torch.float16)

        processor = AutoProcessor.from_pretrained(self.model_id)
        prompt_chunks = self.__get_prompt_chunks(chunk_size=4)

        caption = ""
        with torch.no_grad():
            for prompt_list in prompt_chunks:
                prompt = self.__get_single_answer_prompt(prompt_list)
                inputs = processor(prompt, raw_image, return_tensors="pt").to(
                    model_management.get_torch_device(), torch.float16
                )
                output = model.generate(
                    **inputs, max_new_tokens=self.max_tokens_per_chunk, do_sample=False
                )
                decoded = processor.decode(output[0][2:])
                cleaned = self.clean_output(decoded)
                caption += cleaned

        del model
        torch.cuda.empty_cache()

        return caption

    def clean_output(self, decoded_output, delimiter=","):
        output_only = decoded_output.split("ASSISTANT: ")[1]
        lines = output_only.split("\n")
        cleaned_output = ""
        for line in lines:
            cleaned_output += self.__replace_delimiter(line, ".", delimiter)

        return cleaned_output

    def __get_single_answer_prompt(self, questions):
        """
        For multiple turns conversation:
        "USER: <image>\n<prompt1> ASSISTANT: <answer1></s>USER: <prompt2> ASSISTANT: <answer2></s>USER: <prompt3> ASSISTANT:"
        From: https://huggingface.co/docs/transformers/en/model_doc/llava#usage-tips
        Not sure how the formatting works for multi-turn but those are the docs.
        """
        prompt = "USER: <image>\n"
        for index, question in enumerate(questions):
            if index != 0:
                prompt += "USER: "
            prompt += f"{question} </s >"
        prompt += "ASSISTANT: "

        return prompt

    def __replace_delimiter(self, text: str, old, new=","):
        """Replace only the LAST instance of old with new"""
        if old not in text:
            return text.strip() + " "
        last_old_index = text.rindex(old)
        replaced = text[:last_old_index] + new + text[last_old_index + len(old) :]
        return replaced.strip() + " "

    def __get_prompt_chunks(self, chunk_size=4):
        prompt_chunks = []
        for index, feature in enumerate(self.question_list):
            if index % chunk_size == 0:
                prompt_chunks.append([feature])
            else:
                prompt_chunks[-1].append(feature)
        return prompt_chunks
