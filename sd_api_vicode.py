from __future__ import annotations

import os
import base64
import io
import time
import modules.shared as shared
from modules.shared import opts
from PIL import PngImagePlugin,Image
from modules import timer
from modules import initialize_util
from modules import initialize
from contextlib import closing
startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()

initialize.check_versions()



from fastapi import FastAPI
from modules.shared_cmd_options import cmd_opts

initialize.initialize()

app = FastAPI()
initialize_util.setup_middleware(app)



from modules.shared_cmd_options import cmd_opts
from typing import Any, Dict
from modules.api import models
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images






def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=100)


        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)




class Txt2ImgRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




def gen_img2(txt2imgreq):
    # Initialisation et validation des paramètres
    args = vars(txt2imgreq)
    finished = False

    send_images = True
    # Exécution du processus de génération d'image
    with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:

        try:
            processed = process_images(p)
        finally:
            finished = True

    # Conversion des images en base64 si nécessaire
    #print(processed.images)
    #print("------------")
    b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    
    return b64images


    

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion model.")

    # Define all the arguments that your function can accept
    parser.add_argument("--prompt", type=str, default="realistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))", help="Image prompt")
    parser.add_argument("--negative_prompt", type=str, default="bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, Duplicate, two people, text, signature, watermark", help="Negative prompt to avoid certain features")
    parser.add_argument("--styles", type=str, default=None, help="Styles for the image")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for random number generator")
    parser.add_argument("--subseed", type=int, default=-1, help="Subseed for random number generator")
    parser.add_argument("--subseed_strength", type=int, default=0, help="Subseed strength")
    parser.add_argument("--sampler_name", type=str, default='DPM++ 2M Karras', help="Sampler name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations")
    parser.add_argument("--steps", type=int, default=25, help="Number of steps for image generation")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=768, help="Height of the generated image")
    parser.add_argument("--denoising_strength", type=float, default=0.2, help="Denoising strength")
    parser.add_argument("--enable_hr", type=bool, default=True, help="Enable high resolution")
    parser.add_argument("--hr_scale", type=float, default=1.45, help="High resolution scale")
    parser.add_argument("--hr_upscaler", type=str, default='ESRGAN_4x', help="High resolution upscaler")
    parser.add_argument("--hr_second_pass_steps", type=int, default=20, help="High resolution second pass steps")
    # Continue adding other parameters as needed

    return parser.parse_args()





def main():
    args = parse_arguments()

    # Création du dictionnaire avec les valeurs par défaut des paramètres
    gen_image_settings = {'prompt': 'realistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))', 'negative_prompt': 'bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,Duplicate , two people,text ,signature,watermark', 'styles': None, 'seed': -1, 'subseed': -1, 'subseed_strength': 0, 'seed_resize_from_h': -1, 'seed_resize_from_w': -1, 'sampler_name': 'DPM++ 2M Karras', 'batch_size': 1, 'n_iter': 1, 'steps': 25, 'cfg_scale': 7.5, 'width': 512, 'height': 768, 'restore_faces': None, 'tiling': None, 'do_not_save_samples': True, 'do_not_save_grid': True, 'eta': None, 'denoising_strength': 0.2, 's_min_uncond': None, 's_churn': None, 's_tmax': None, 's_tmin': None, 's_noise': None, 'override_settings': {'sd_model_checkpoint': 'v1-5-pruned-emaonly'}, 'override_settings_restore_afterwards': True, 'refiner_checkpoint': None, 'refiner_switch_at': None, 'disable_extra_networks': False, 'comments': None, 'enable_hr': True, 'firstphase_width': 0, 'firstphase_height': 0, 'hr_scale': 1.45, 'hr_upscaler': 'ESRGAN_4x', 'hr_second_pass_steps': 20, 'hr_resize_x': 0, 'hr_resize_y': 0, 'hr_checkpoint_name': 'v1-5-pruned-emaonly', 'hr_sampler_name': 'DPM++ 2M Karras', 'hr_prompt': '', 'hr_negative_prompt': '', 'sampler_index': None}

    # Call your gen_img2 function with these settings
    image_result = gen_img2(Txt2ImgRequest(**gen_image_settings))
    print (image_result)

if __name__ == "__main__":
    main()
