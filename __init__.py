from .src.img2txt_node import Img2TxtNode

_node_name = "img2txt BLIP/Llava Multimodel Tagger"

NODE_CLASS_MAPPINGS = {
    _node_name: Img2TxtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {_node_name: "Image to Text - Auto Caption"}

WEB_DIRECTORY = "./web"
