from typing import Sequence, Mapping, Any, Union
import random
import torch
import boto3
import os, sys
import warnings
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

warnings.filterwarnings("ignore")

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db as db

COMFY_PATH = os.getenv('COMFY_PATH', '/app/ComfyUI')
sys.path.append(COMFY_PATH)

IS_DOCKER = os.getenv('IS_DOCKER', 'false').lower() == 'true'

s3 = boto3.client("s3")

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
    # keyword = vs.summarize_title_in_word(name)
    print(f"* Title: {name}")
    name = vs.rephrase_title(name, model="claude-3-5-sonnet-20240620")
    caption = (
        f'"{name}", "tarot and computers collection", stunning award-winning pixel art'
    )
    print("--> " + caption)

    # Force CPU usage if not in Docker
    if not IS_DOCKER:
        device = torch.device("cpu")
        torch.set_default_tensor_type(torch.FloatTensor)

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
            text="low quality, glitch, blurry, deformed, mutated, ugly, disfigured, grainy, noise,"
            "watermark, cartoon, anime, videogame, text, flow chart, signature, depth of field,"
            "religious, portrait, profile, mandala, photo-real, b&w, poker, modern, grid, 3d, knolling",
            clip=get_value_at_index(loraloader_39, 1),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_40 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        vaeloader = VAELoader()
        vaeloader_48 = vaeloader.load_vae(vae_name="sdxl.vae.safetensors")

        cliptextencode_102 = cliptextencode.encode(
            text=caption,
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
                sampler_name="dpmpp_2m_sde_gpu" if IS_DOCKER else "dpmpp_2m_sde",
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
    vs.validate_openai_env()
    arxiv_codes = db.get_arxiv_id_list(pu.db_params, "summaries")
    title_dict = db.get_arxiv_title_dict(pu.db_params)
    img_dir = os.path.join(PROJECT_PATH, "imgs/")

    done_imgs = pu.list_s3_files("llmpedia", strip_extension=True)
    arxiv_codes = list(set(arxiv_codes) - set(done_imgs))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for idx, arxiv_code in enumerate(arxiv_codes):
        name = title_dict[arxiv_code]
        img_file = img_dir + arxiv_code + ".png"
        clean_name = (
            name
            # .replace("transformer", "processor")
            .replace("Transformer", "Machine")
            # .replace("Matrix", "Linear Algebra")
            .replace("Large Language Model", "LLM").replace("LLM", "Model")
        )
        generate_image(clean_name, img_file)

        ## Upload to s3.
        s3.upload_file(img_file, "llmpedia", arxiv_code + ".png")
        print(f"Saved {img_file} ({idx+1}/{len(arxiv_codes)})")


if __name__ == "__main__":
    main()
