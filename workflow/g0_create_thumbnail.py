from typing import Sequence, Mapping, Any, Union
from enum import Enum
import random
import torch
import base64
import requests
import boto3
import time
import os, sys
import warnings
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

warnings.filterwarnings("ignore")

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "g0_create_thumbnail.log")

# COMFY_PATH = os.getenv('COMFY_PATH', '/app/ComfyUI')
# sys.path.append(COMFY_PATH)

# IS_DOCKER = os.getenv('DOCKER_CONTAINER', 'true').lower() == 'true'

# if IS_DOCKER:
#     from utils.cpu_override import apply_overrides, modify_comfy_model_management
#     modify_comfy_model_management()
#     apply_overrides()

s3 = boto3.client("s3")

# from nodes import (
#     KSampler,
#     LoraLoader,
#     CLIPTextEncode,
#     EmptyLatentImage,
#     VAELoader,
#     VAEDecode,
#     ImageScaleBy,
#     CheckpointLoaderSimple,
#     SaveImage,
# )


class RDModel(str, Enum):
    RD_CLASSIC = "RD_CLASSIC"
    RD_FLUX = "RD_FLUX"


def call_rd_api(
    api_token: str,
    prompt: str,
    width: int = 128,
    height: int = 128,
    model: RDModel = RDModel.RD_FLUX,
    steps: int = 20,
    num_images: int = 1,
    seed: int = 424242,
    prompt_style: str = "default",
    input_image: str = None,
) -> dict:
    """Generate an image using the RetroDiffusion API"""

    response = requests.post(
        "https://api.retrodiffusion.ai/v1/inferences",
        headers={"X-RD-Token": api_token},
        json={
            "prompt": prompt,
            "width": width,
            "height": height,
            "model": model,
            "steps": steps,
            "num_images": num_images,
            "seed": seed,
            "prompt_style": prompt_style,
            **({"input_image": input_image} if input_image else {}),
        },
    )
    response.raise_for_status()
    return response.json()


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def generate_image(title: str, img_file: str) -> None:
    """Generate a thumbnail image for a paper."""
    title = vs.rephrase_title(title, model="claude-3-5-sonnet-20241022")
    caption = (
        f'"{title}", "tarot and computers collection", stunning award-winning classic pixel art'
    )
    logger.info(f"--> Generated caption: {caption}")
    for attempt in range(3):
        try:
            result = call_rd_api(
                api_token=os.getenv("RD_API_KEY"),
                prompt=caption,
            )
            break
        except requests.exceptions.HTTPError as e:
            if attempt == 2:
                raise
            logger.warning(f"API call failed (attempt {attempt + 1}/3), retrying...")
            time.sleep(10)

    b64_img = result['base64_images'][0]
    
    with open(img_file, "wb") as f:
        f.write(base64.b64decode(b64_img))

    logger.info(f"--> Saved image to: {img_file}")

    # with torch.inference_mode():
    #     checkpointloadersimple = CheckpointLoaderSimple()
    #     checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
    #         ckpt_name="sd_xl_base_1.0.safetensors"
    #     )

    #     loraloader = LoraLoader()
    #     loraloader_39 = loraloader.load_lora(
    #         lora_name="pixel-art-xl.safetensors",
    #         strength_model=1,
    #         strength_clip=1,
    #         model=get_value_at_index(checkpointloadersimple_4, 0),
    #         clip=get_value_at_index(checkpointloadersimple_4, 1),
    #     )

    #     cliptextencode = CLIPTextEncode()
    #     cliptextencode_7 = cliptextencode.encode(
    #         text="low quality, glitch, blurry, deformed, mutated, ugly, disfigured, grainy, noise,"
    #         "watermark, cartoon, anime, videogame, text, flow chart, signature, depth of field,"
    #         "religious, portrait, profile, mandala, photo-real, b&w, poker, modern, grid, 3d, knolling",
    #         clip=get_value_at_index(loraloader_39, 1),
    #     )

    #     emptylatentimage = EmptyLatentImage()
    #     emptylatentimage_40 = emptylatentimage.generate(
    #         width=1024, height=1024, batch_size=1
    #     )

    #     vaeloader = VAELoader()
    #     vaeloader_48 = vaeloader.load_vae(vae_name="sdxl.vae.safetensors")

    #     cliptextencode_102 = cliptextencode.encode(
    #         text=caption,
    #         clip=get_value_at_index(loraloader_39, 1),
    #     )

    #     ksampler = KSampler()
    #     vaedecode = VAEDecode()
    #     imagescaleby = ImageScaleBy()
    #     saveimage = SaveImage(output_dir=PROJECT_PATH)

    #     for q in range(10):
    #         ksampler_103 = ksampler.sample(
    #             seed=random.randint(1, 2**64),
    #             steps=20,
    #             cfg=8,
    #             sampler_name="dpmpp_2m_sde_gpu" if IS_DOCKER else "dpmpp_2m_sde",
    #             scheduler="karras",
    #             denoise=1,
    #             model=get_value_at_index(loraloader_39, 0),
    #             positive=get_value_at_index(cliptextencode_102, 0),
    #             negative=get_value_at_index(cliptextencode_7, 0),
    #             latent_image=get_value_at_index(emptylatentimage_40, 0),
    #         )

    #         vaedecode_25 = vaedecode.decode(
    #             samples=get_value_at_index(ksampler_103, 0),
    #             vae=get_value_at_index(vaeloader_48, 0),
    #         )

    #         imagescaleby_42 = imagescaleby.upscale(
    #             upscale_method="nearest-exact",
    #             scale_by=0.125,
    #             image=get_value_at_index(vaedecode_25, 0),
    #         )

    #         imagescaleby_45 = imagescaleby.upscale(
    #             upscale_method="nearest-exact",
    #             scale_by=8,
    #             image=get_value_at_index(imagescaleby_42, 0),
    #         )

    #         saveimage_104 = saveimage.save_images(
    #             filename_prefix=img_file.replace(".png", ""),
    #             images=get_value_at_index(imagescaleby_45, 0),
    #         )

    #         return True


def main():
    logger.info("Starting thumbnail creation process")
    ## Load the mapping files.
    vs.validate_openai_env()
    arxiv_codes = db_utils.get_arxiv_id_list("summaries")
    title_dict = db_utils.get_arxiv_title_dict()
    img_dir = os.path.join(PROJECT_PATH, "data", "arxiv_art/")

    done_imgs = pu.list_s3_files("arxiv-art", strip_extension=True)
    arxiv_codes = list(set(arxiv_codes) - set(done_imgs))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to process for thumbnails.")

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_title = title_dict[arxiv_code]
        logger.info(f"[{idx}/{total_papers}] Creating thumbnail: {arxiv_code} - '{paper_title}'")
        
        img_file = img_dir + arxiv_code + ".png"
        clean_name = (
            paper_title.replace("Transformer", "Machine")
            .replace("Large Language Model", "LLM")
            .replace("LLM", "Model")
        )
        generate_image(clean_name, img_file)

        ## Upload to s3.
        s3.upload_file(img_file, "arxiv-art", arxiv_code + ".png")
        logger.info(f"[{idx}/{total_papers}] Uploaded thumbnail: {arxiv_code} - '{paper_title}'")

    logger.info("Thumbnail creation process completed.")


if __name__ == "__main__":
    main()
