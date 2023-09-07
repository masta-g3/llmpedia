from typing import Sequence, Mapping, Any, Union
import random
import torch
import json
import os, sys
import warnings
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
COMFY_PATH = os.environ.get("COMFY_PATH")
sys.path.append(COMFY_PATH)
warnings.filterwarnings("ignore")

import utils.paper_utils as pu

from nodes import (
    KSampler,
    LoraLoader,
    CLIPTextEncode,
    EmptyLatentImage,
    VAELoader,
    VAEDecode,
    ImageScaleBy,
    CheckpointLoaderSimple,
    SaveImage,
)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def generate_image(name, img_file):
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_39 = loraloader.load_lora(
            lora_name="pixel-art-xl.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_7 = cliptextencode.encode(
            text="low quality, ugly, distorted, blurry, deformed, watermark, " +
                  "text, flow chart, signature, depth of field, " +
                  "mandala, star map, photoreal, b&w, poker, modern, grainy",
            clip=get_value_at_index(loraloader_39, 1),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_40 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        vaeloader = VAELoader()
        vaeloader_48 = vaeloader.load_vae(vae_name="sdxl.vae.safetensors")

        cliptextencode_102 = cliptextencode.encode(
            text=f'"{name}", tarot and computers',
            clip=get_value_at_index(loraloader_39, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        imagescaleby = ImageScaleBy()
        saveimage = SaveImage(output_dir=PROJECT_PATH)

        for q in range(10):
            ksampler_103 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="dpmpp_2m_sde_gpu",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(loraloader_39, 0),
                positive=get_value_at_index(cliptextencode_102, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_40, 0),
            )

            vaedecode_25 = vaedecode.decode(
                samples=get_value_at_index(ksampler_103, 0),
                vae=get_value_at_index(vaeloader_48, 0),
            )

            imagescaleby_42 = imagescaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=0.125,
                image=get_value_at_index(vaedecode_25, 0),
            )

            imagescaleby_45 = imagescaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=8,
                image=get_value_at_index(imagescaleby_42, 0),
            )

            saveimage_104 = saveimage.save_images(
                filename_prefix=img_file.replace(".png", ""),
                images=get_value_at_index(imagescaleby_45, 0),
            )

            return True


def main():
    ## Load the mapping files.
    title_dict = pu.get_arxiv_title_dict(pu.db_params)
    img_dir = os.path.join(PROJECT_PATH, "imgs/")

    for idx, (code, name) in enumerate(title_dict.items()):
        img_file = img_dir + code + ".png"
        if os.path.exists(img_file):
            continue
        else:
            generate_image(name, img_file)
            print(f"Saved {img_file} ({idx+1}/{len(title_dict)})")


if __name__ == "__main__":
    main()
