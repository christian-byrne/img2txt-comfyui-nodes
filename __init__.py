from .src.img2txt_node import Img2TxtNode

NODE_CLASS_MAPPINGS = {
    "img2txt BLIP/Llava Multimodel Tagger": Img2TxtNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "img2txt BLIP/Llava Multimodel Tagger": "Image to Text - Auto Caption"
}
WEB_DIRECTORY = "./web"
