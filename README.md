

**Auto-generate caption (BLIP Only)**:

![alt text](wiki/demo-pics/Selection_003.png)

**Using to automate img2img process (BLIP and Llava)**

![alt text](wiki/demo-pics/Selection_002.png)


## Requirements/Dependencies

- Pillow==10.0.0
- Pillow==10.3.0
- torch==2.2.1
- torchvision==0.17.1
- transformers==4.39.2

## Installation


- `cd` into ComfyUI/custom_nodes directory
- `git clone` this repo
- `cd` into the repo
- `python3 -m pip install -r requirements.txt`
- Models will be automatically downloaded per-use. If you never toggle a model on in the UI, it will never be downloaded. Don't toggle on the Llava model if you don't want to download 15Gb. Outputs with BLIP only are still very good and only 1Gb w/ fast inference.


## Models Implemented (so far)

- [Salesforce - blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - **Title**: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation 
  - **Size**: ~ 1GB
  - **Dataset**: COCO (The MS COCO dataset is a large-scale object detection, image segmentation, and captioning dataset published by Microsoft)
- [llava - llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - **Title**: LLava: Large Language Models for Vision and Language Tasks 
  - **Size**: ~ 15GB
  - **Dataset**: 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP, 158K GPT-generated multimodal instruction-following data, 450K academic-task-oriented VQA data mixture, 40K ShareGPT data.
  - **Notable Requirements**: transformers >= 4.35.3
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


