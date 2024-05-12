from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LlavaImg2Txt:
    def __init__(self):
        self.model_id = "llava-hf/llava-1.5-7b-hf"

        self.ideal_prompt_features = {
            "subject": [
                "What is the the subject of this?",
            ],
            "medium": [
                "What are the mediums used to make this?",
            ],
            "style": [
                "What are the the artistic styles this is reminiscent of?",
            ],
            "mood": [
                "What is the mood of this?",
            ],
            "artists": [
                "Which famous artists is this reminiscent of?",
            ],
            "website": [
                "What art website was this most likely found on?",
            ],
            "resolution": [
                "How sharp or detailed is this image?",
            ],
            "environment": [
                "What is the environment and background of this image?",
            ],
            "location": [
                "What is the location of this image?",
            ],
            "details and objects": [
                "What are the objects in this image?",
            ],
            "colors": [
                "What is the color palette in this image?",
            ],
            "lighting": [
                "What is the lighting in this image?",
            ],
        }

    def generate_caption(self, image: Image):
        # Convert Image to RGB first
        if image.getmodebase() != "RGB":
            image = image.convert("RGB")

        caption, raw_caption = self.__get_caption(image)
        return caption

    def clean_output(self, decoded_output, delimiter=","):
        output_only = decoded_output.split("ASSISTANT: ")[1]
        lines = output_only.split("\n")
        cleaned_output = ""
        split_candidates = [
            "reminiscent of include ",
            "reminiscent of includes ",
            "reminiscent of an ",
            "reminiscent of a ",
            "reminiscent of ",
            "giving the scene of ",
            "giving the scene a ",
            "appears to be ",
            "includes a ",
            "includes an ",
            "features a ",
            "image is ",
            "this is an ",
            "this is an ",
            "this is a ",
            "this is ",
            "capturing the ",
            "image of ",
        ]
        for line in lines:
            if line != "":
                # found_split = False
                for candidate in split_candidates:
                    if candidate in line:
                        line = line.split(candidate)[-1:][0]
                        # cleaned_output += replace_delimiter(
                        #     line.split(candidate)[-1:][0], ".", delimiter
                        # )

                        # found_split = True

                cleaned_output += self.__replace_delimiter(line, ".", delimiter)

        return cleaned_output

    def __get_single_answer_prompt(self, prompt_features):
        """
        For multiple turns conversation:
        "USER: <image>\n<prompt1> ASSISTANT: <answer1></s>USER: <prompt2> ASSISTANT: <answer2></s>USER: <prompt3> ASSISTANT:"
        From: https://huggingface.co/docs/transformers/en/model_doc/llava#usage-tips
        Not sure how the formatting works for multi-turn but those are the docs.

        """
        prompt = "USER: <image>\n"
        for index, feature in enumerate(prompt_features):
            if index != 0:
                prompt += "USER: "
            prompt += f"{feature} </s >"
        prompt += "ASSISTANT: "

        return prompt

    def __replace_delimiter(self, text: str, old, new=","):
        """Replace only the LAST instance of old with new"""
        if old not in text:
            return text.strip() + " "
        last_old_index = text.rindex(old)
        replaced = text[:last_old_index] + new + text[last_old_index + len(old) :]
        return replaced.strip() + " "

    def __get_prompt_chunks(self, prompt_features, chunk_size=4):
        prompt_chunks = []
        for index, feature in enumerate(prompt_features):
            if index % chunk_size == 0:
                prompt_chunks.append([])
            prompt_chunks[-1].append(prompt_features[feature][0])
        return prompt_chunks

    def __get_caption(
        self,
        raw_image: Image,
        use_4bit_quantization=True,
        use_flash2_attention=False,
        use_low_cpu_mem_usage=True,
    ):
        """

        Args:
            prompt (str): The prompt to generate the caption for
            use_4bit_quantization (bool): Whether to use 4-bit quantization to reduce memory usage. 4-bit quantization reduces the precision of model parameters, potentially affecting the quality of generated outputs. Use if VRAM is limited.
            use_flash2_attention (bool): Whether to use Flash-Attention 2. Flash-Attention 2 focuses on optimizing attention mechanisms, which are crucial for the model's performance during generation. Use if computational resources are abundant.
            use_low_cpu_mem_usage (bool): In low_cpu_mem_usage mode, the model uses less memory on CPU, which can be useful when running multiple models on the same CPU. The model is initialized with optimizations aimed at reducing CPU memory consumption. This can be beneficial when working with large models or limited computational resources, such as systems with low RAM.
        """
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=use_low_cpu_mem_usage,
            load_in_4bit=use_4bit_quantization,
            use_flash_attention_2=use_flash2_attention,
        )

        # model.to() is not supported for 4-bit or 8-bit bitsandbytes models. With 4-bit quantization, use the model as it is, since the model will already be set to the correct devices and casted to the correct `dtype`.
        if torch.cuda.is_available() and not use_4bit_quantization:
            model = model.to(0)

        processor = AutoProcessor.from_pretrained(self.model_id)
        prompt_chunks = self.__get_prompt_chunks(
            self.ideal_prompt_features, chunk_size=4
        )

        caption = ""
        raw_caption = ""
        for prompt_list in prompt_chunks:
            prompt = self.__get_single_answer_prompt(prompt_list)
            inputs = processor(prompt, raw_image, return_tensors="pt").to(
                0, torch.float16
            )
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            decoded = processor.decode(output[0][2:], skip_special_tokens=True)
            raw_caption += decoded
            cleaned = self.clean_output(decoded)
            caption += cleaned

        return caption, raw_caption


if __name__ == "__main__":
    x = LlavaImg2Txt()
    test_pic_path = "/home/c_byrne/projects/comfy-node-testing-tools/data/test-images/rgb/movie-scene/rgb-movie-scene-H576px_W1024px-jpg-26.jpg"
    test_image = Image.open(test_pic_path).convert("RGB")
