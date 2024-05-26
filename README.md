

**Auto-generate caption (BLIP Only)**:

![alt text](wiki/demo-pics/Selection_003.png)

**Using to automate img2img process (BLIP and Llava)**

![alt text](wiki/demo-pics/Selection_002.png)


## Requirements/Dependencies

- Shared with ComfyUI
  - Pillow>=8.3.2
  - torch>=2.2.1
  - torchvision>=0.17.1
- For Llava model
  - bitsandbytes>=0.43.0
  - accelerate>=0.3.0
- For MiniCPM model
  - transformers>=4.36.0
  - timm==0.9.10
  - sentencepiece==0.1.99
- Python 3.10+

## Installation


- `cd` into `ComfyUI/custom_nodes` directory
- `git clone` this repo
- `cd img2txt-comfyui-nodes`
- `pip install -r requirements.txt`
- Models will be automatically downloaded per-use. If you never toggle a model on in the UI, it will never be downloaded.
- To ask a list of specific questions about the image, use the Llava or MiniPCM models. The questions are separated by line in the multiline text input box.

## Support for Chinese

- The `MiniCPM` model works with Chinese text input without any additional configuration. The output will also be in Chinese. 
  - "MiniCPM-V 2.0 supports strong bilingual multimodal capabilities in both English and Chinese. This is enabled by generalizing multimodal capabilities across languages, a technique from VisCPM"
<!-- - Here are the input field descriptions in Chinese, translated by  -->

## Tips

- The multi-line input can be used to ask any type of questions. You can even ask very specific or complex questions about images.
- To get best results for a prompt that will be fed back into a txt2img or img2img prompt, usually it's best to only ask one or two questions, asking for a general description of the image and the most salient features and styles.

## Model Locations/Paths

- Models are downloaded automatically using the Huggingface cache system and the transformers `from_pretrained` method so no manual installation of models is necessary.
- If you really want to manually download the models, please refer to [Huggingface's documentation concerning the cache system](https://huggingface.co/docs/transformers/main/en/installation#cache-setup). Here is the relevant except:
  - Pretrained models are downloaded and locally cached at  `~/.cache/huggingface/hub`. This is the default directory given by the shell environment variable TRANSFORMERS_CACHE. On Windows, the default directory is given by `C:\Users\username\.cache\huggingface\hub`. You can change the shell environment variables shown below - in order of priority - to specify a different cache directory:
    - Shell environment variable (default): HUGGINGFACE_HUB_CACHE or TRANSFORMERS_CACHE.
    - Shell environment variable: HF_HOME.
    - Shell environment variable: XDG_CACHE_HOME + /huggingface.


## Models Implemented (so far)

- [MiniCPM](https://huggingface.co/openbmb/MiniCPM-V-2/tree/main) (Chinese & English)
  - **Title**: MiniCPM-V-2 - Strong multimodal large language model for efficient end-side deployment
  - **Datasets**: HuggingFaceM4VQAv2, RLHF-V-Dataset, LLaVA-Instruct-150K
  - **Size**: ~ 6.8GB
- [Salesforce - blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - **Title**: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation 
  - **Size**: ~ 2GB
  - **Dataset**: COCO (The MS COCO dataset is a large-scale object detection, image segmentation, and captioning dataset published by Microsoft)
- [llava - llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - **Title**: LLava: Large Language Models for Vision and Language Tasks 
  - **Size**: ~ 15GB
  - **Dataset**: 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP, 158K GPT-generated multimodal instruction-following data, 450K academic-task-oriented VQA data mixture, 40K ShareGPT data.
- Coming Soon: [Microsoft - Git Large coco](https://huggingface.co/microsoft/git-large-coco)
  - **Title**: GIT (short for GenerativeImage2Text) mode
  - **Size**: ~ 3GB
  - **Dataset**: COCO  
- [More - to do](https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending)

## Prompts


This is the guide for the format of an "ideal" txt2img prompt (using BLIP). Use as the basis for the questions to ask the img2txt models.

- **Subject** - you can specify region, write the most about the subject
- **Medium** - material used to make artwork. Some examples are illustration, oil painting, 3D rendering, and photography. Medium has a strong effect because one keyword alone can dramatically change the style.
- **Style** - artistic style of the image. Examples include impressionist, surrealist, pop art, etc.
- **Artists**  - Artist names are strong modifiers. They allow you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles. Now let’s add Stanley Artgerm Lau, a superhero comic artist, and Alphonse Mucha, a portrait painter in the 19th century.
- **Website** - Niche graphic websites such as Artstation and Deviant Art aggregate many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles.
- **Resolution** - Resolution represents how sharp and detailed the image is. Let’s add keywords highly detailed and sharp focus
- **Enviornment**
- **Additional** Details and objects - Additional details are sweeteners added to modify an image. We will add sci-fi, stunningly beautiful and dystopian to add some vibe to the image.
- **Composition** - camera type, detail, cinematography, blur, depth-of-field
- **Color/Warmth** - You can control the overall color of the image by adding color keywords. The colors you specified may appear as a tone or in objects.
- **Lighting** - Any photographer would tell you lighting is a key factor in creating successful images. Lighting keywords can have a huge effect on how the image looks. Let’s add cinematic lighting and dark to the prompt.


