
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

## Prompts

Where possible, attempt to get descriptions to follow the format that works well for image generation (based on the assumption that you want to use img2txt as part of workflow in which you use one image or generation as the prompt for another — or something along those lines).

This is the guide for the format of an "ideal" txt2img prompt (using BLIP):

- **Subject** - you can specify region, write the most about the subject
- **Medium** - material used to make artwork. Some examples are illustration, oil painting, 3D rendering, and photography. Medium has a strong effect because one keyword alone can dramatically change the style.
- **Style** - artistic style of the image. Examples include impressionist, surrealist, pop art, etc.
- **Artists**  - Artist names are strong modifiers. They allow you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles. Now let’s add Stanley Artgerm Lau, a superhero comic artist, and Alphonse Mucha, a portrait painter in the 19th century.
<!-- - **Website** - Niche graphic websites such as Artstation and Deviant Art aggregate many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles. -->
- **Resolution** - Resolution represents how sharp and detailed the image is. Let’s add keywords highly detailed and sharp focus
- **Enviornment**
- **Additional** Details and objects - Additional details are sweeteners added to modify an image. We will add sci-fi, stunningly beautiful and dystopian to add some vibe to the image.
- **Composition** - camera type, detail, cinematography, blur, depth-of-field
- **Color/Warmth** - You can control the overall color of the image by adding color keywords. The colors you specified may appear as a tone or in objects.
- **Lighting** - Any photographer would tell you lighting is a key factor in creating successful images. Lighting keywords can have a huge effect on how the image looks. Let’s add cinematic lighting and dark to the prompt.


