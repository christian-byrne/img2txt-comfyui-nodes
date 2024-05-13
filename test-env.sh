#!/bin/bash

default_python_version="3.10"
default_dir_name="temp-test_ev"

if [ -z "$1" ]; then
    python_version="$default_python_version"
else
    python_version="$1"
fi

if [ -z "$2" ]; then
    dir_name="$default_dir_name"
else
    dir_name="$2"
fi

if [ -d "$dir_name" ]; then
    echo "Directory $dir_name already exists. Please remove it and try again."
    exit 1
fi

mkdir $dir_name
cd $dir_name

if [ -z "$(command -v pyenv)" ]; then
    echo "pyenv not found. Please install it and try again."
    exit 1
fi

if [ -z "$(command -v git)" ]; then
    echo "git not found. Please install it and try again."
    exit 1
fi

if pyenv versions | grep -q $python_version; then
    echo "Python $python_version is already installed."
else
    pyenv install $python_version
fi
pyenv local $python_version

python -m venv venv
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
cp ~/tools/sd/sd-interfaces/stable-diffusion-webui/models/Stable-diffusion/inpainting/experience_70-inpainting.safetensors ComfyUI/models/checkpoints

py_ver_no_patch_num=$(echo $python_version | cut -d '.' -f 1,2)
# py_ver_no_patch_num will look like "3.10"

# export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib64/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python$py_ver_no_patch_num/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

xdg-open http://localhost:8188 & disown
cd ComfyUI
python main.py

