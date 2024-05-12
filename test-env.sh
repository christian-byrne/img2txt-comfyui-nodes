#!/bin/bash

mkdir test-temp
cd test-temp

python3 -m venv venv
source venv/bin/activate

git clone git@github.com:comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

cd custom_nodes
git clone git@github.com:christian-byrne/img2txt-comfyui-nodes.git
cd img2txt-comfyui-nodes
pip install -r requirements.txt
cd ../../..

cp ~/tools/sd/sd-interfaces/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors ComfyUI/models/checkpoints
cp /home/c_byrne/tools/sd/sd-interfaces/stable-diffusion-webui/models/Stable-diffusion/inpainting/experience_70-inpainting.safetensors ComfyUI/models/checkpoints

export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib64/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

xdg-open http://localhost:8188 & disown
cd ComfyUI
python main.py

