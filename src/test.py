import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"


ideal_prompt_features = {
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
}


def replace_delimiter(text: str, old, new=","):
    """Replace only the LAST instance of old with new"""
    if old not in text:
        return text.strip() + " "
    last_old_index = text.rindex(old)
    replaced = text[:last_old_index] + new + text[last_old_index + len(old) :]
    return replaced.strip() + " "


def clean_output(decoded_output, delimiter=","):
    print(f"Decoded Output: {decoded_output}")
    output_only = decoded_output.split("ASSISTANT: ")[1]
    lines = output_only.split("\n")
    cleaned_output = ""
    split_candidates = [
        "image is ",
        "this is an ",
        "this is a ",
        "this is ",
        "image of ",
    ]
    for line in lines:
        if line != "":
            found_split = False
            for candidate in split_candidates:
                if candidate in line:
                    cleaned_output += replace_delimiter(
                        line.split(candidate)[-1:][0], ".", delimiter
                    )

                    found_split = True
                    break
            if not found_split:
                cleaned_output += replace_delimiter(line, ".", delimiter)

    print(f"Cleaned Output: {cleaned_output}")


def get_single_answer_prompt(prompt_features, question_count=4):
    """
    For multiple turns conversation:
    "USER: <image>\n<prompt1> ASSISTANT: <answer1></s>USER: <prompt2> ASSISTANT: <answer2></s>USER: <prompt3> ASSISTANT:"
    From: https://huggingface.co/docs/transformers/en/model_doc/llava#usage-tips
    Not sure how the formatting works for multi-turn but those are the docs.

    """
    prompt = "USER: <image>\n"
    for index, feature in enumerate(prompt_features):
        if index >= question_count:
            break
        if index != 0:
            prompt += "USER: "
        prompt += f"{prompt_features[feature][0]} </s>"
    prompt += "ASSISTANT: "
    return prompt


# pic_path = "/media/c_byrne/Seagate Expansion Drive/eastside/nudify-tools/spanish-onlyfans-1/inessa-chimato_0024.jpg"

pic_path = "/home/c_byrne/projects/comfy-node-testing-tools/data/test-images/rgb/movie-scene/rgb-movie-scene-H576px_W1024px-jpg-26.jpg"
prompt = "USER: <image>\nWhat is the subject of this? </s>USER: What are the mediums used to make this? </s>USER: What are the artistic styles this is reminiscent of? </s>USER: What is the mood of this? </s>ASSISTANT: "

prompt = get_single_answer_prompt(ideal_prompt_features)
print(prompt)

# Whether to use 4-bit quantization to reduce memory usage. 4-bit quantization reduces the precision of model parameters, potentially affecting the quality of generated outputs. Use if VRAM is limited.
use_4bit_quantization = True
# Whether to use Flash-Attention 2. Flash-Attention 2 focuses on optimizing attention mechanisms, which are crucial for the model's performance during generation. Use if computational resources are abundant.
use_flash2_attention = False
# In low_cpu_mem_usage mode, the model uses less memory on CPU, which can be useful when running multiple models on the same CPU. The model is initialized with optimizations aimed at reducing CPU memory consumption. This can be beneficial when working with large models or limited computational resources, such as systems with low RAM.
use_low_cpu_mem_usage = True

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_4bit=use_4bit_quantization,
    low_cpu_mem_usage=use_low_cpu_mem_usage,
    use_flash_attention_2=use_flash2_attention,
)

# model.to() is not supported for 4-bit or 8-bit bitsandbytes models. With 4-bit quantization, use the model as it is, since the model will already be set to the correct devices and casted to the correct `dtype`.
if torch.cuda.is_available() and not use_4bit_quantization:
    model = model.to(0)


processor = AutoProcessor.from_pretrained(model_id)


raw_image = Image.open(pic_path).convert("RGB")
inputs = processor(prompt, raw_image, return_tensors="pt").to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

decoded = processor.decode(output[0][2:], skip_special_tokens=True)

clean_output(decoded)
