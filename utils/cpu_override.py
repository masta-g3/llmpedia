import torch
import comfy.model_management
import nodes
import folder_paths
from PIL import Image
import numpy as np
import os
import json
from PIL.PngImagePlugin import PngInfo

def get_torch_device():
    return torch.device("cpu")

class SaveImage(nodes.SaveImage):
    def __init__(self, output_dir=None):
        super().__init__()
        self.output_dir = output_dir or folder_paths.get_output_directory()

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = self._create_metadata(prompt, extra_pnginfo)
            file = f"{filename.replace('%batch_num%', str(batch_number))}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
        return {"ui": {"images": results}}

    def _create_metadata(self, prompt, extra_pnginfo):
        if getattr(nodes, 'args', None) and nodes.args.disable_metadata:
            return None
        metadata = PngInfo()
        if prompt:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        return metadata

def apply_overrides():
    comfy.model_management.get_torch_device = get_torch_device
    torch.cuda.is_available = lambda: False
    torch.cuda.current_device = lambda: -1
    torch.cuda.device_count = lambda: 0
    nodes.SaveImage = SaveImage
    print("CPU and SaveImage overrides applied successfully.")