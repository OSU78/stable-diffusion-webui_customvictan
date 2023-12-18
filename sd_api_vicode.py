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

    send_images = True
    # Exécution du processus de génération d'image
    with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:

        try:
            processed = process_images(p)
        finally:
            print("OK nice")

    # Conversion des images en base64 si nécessaire
    print(processed.images)
    print("------------")
    b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    print(b64images)
    return b64images



def gen_image(prompt: str ="ealistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))", negative_prompt: str="bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,Duplicate , two people,text ,signature,watermark", styles: Any = None, seed: int = -1, 
              subseed: int = -1, subseed_strength: int = 0, seed_resize_from_h: int = -1, 
              seed_resize_from_w: int = -1, sampler_name: str = 'DPM++ 2M Karras', 
              batch_size: int = 1, n_iter: int = 1, steps: int = 25, cfg_scale: float = 7.5, 
              width: int = 512, height: int = 768, restore_faces: Any = None, tiling: Any = None, 
              do_not_save_samples: bool = False, do_not_save_grid: bool = False, eta: Any = None, 
              denoising_strength: float = 0.2, s_min_uncond: Any = None, s_churn: Any = None, 
              s_tmax: Any = None, s_tmin: Any = None, s_noise: Any = None, 
              override_settings: Dict[str, Any] = {'sd_model_checkpoint': 'Dautless'}, 
              override_settings_restore_afterwards: bool = True, refiner_checkpoint: Any = None, 
              refiner_switch_at: Any = None, disable_extra_networks: bool = False, 
              comments: Any = None, enable_hr: bool = True, firstphase_width: int = 0, 
              firstphase_height: int = 0, hr_scale: float = 1.45, hr_upscaler: str = 'ESRGAN_4x', 
              hr_second_pass_steps: int = 20, hr_resize_x: int = 0, hr_resize_y: int = 0, 
              hr_checkpoint_name: str = 'Dautless', hr_sampler_name: str = 'DPM++ 2M Karras', 
              hr_prompt: str = '', hr_negative_prompt: str = '', sampler_index: str = 'Euler', 
              script_name: Any = None, script_args: list = [], alwayson_scripts: dict = {}):
    
  # Créer l'instance de traitement d'image
        
    print(models.StableDiffusionTxt2ImgProcessingAPI)
  

    print("DATA")
    print(" ")
    #print(data)
    print(" ----------------------------")
    gen_img2(Txt2ImgRequest(**gen_image_settings))
    
    
    


import argparse


from modules.cmd_args import parser
def main():
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    # Extraire uniquement les clés spécifiques
    keys_to_extract = ["prompt", "negative_prompt", "styles", "seed", "subseed", 
                       "subseed_strength", "sampler_name", "batch_size", "n_iter", 
                       "steps", "cfg_scale", "width", "height", "denoising_strength", 
                       "enable_hr", "hr_scale", "hr_upscaler", "hr_second_pass_steps"]
    specific_args = {key: args_dict[key] for key in keys_to_extract}

    # Création du dictionnaire avec les valeurs par défaut des paramètres
    gen_image_settings = {'prompt': 'realistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))', 'negative_prompt': 'bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,Duplicate , two people,text ,signature,watermark', 'styles': None, 'seed': -1, 'subseed': -1, 'subseed_strength': 0, 'seed_resize_from_h': -1, 'seed_resize_from_w': -1, 'sampler_name': 'DPM++ 2M Karras', 'batch_size': 1, 'n_iter': 1, 'steps': 25, 'cfg_scale': 7.5, 'width': 512, 'height': 768, 'restore_faces': None, 'tiling': None, 'do_not_save_samples': True, 'do_not_save_grid': True, 'eta': None, 'denoising_strength': 0.2, 's_min_uncond': None, 's_churn': None, 's_tmax': None, 's_tmin': None, 's_noise': None, 'override_settings': {'sd_model_checkpoint': 'Dautless'}, 'override_settings_restore_afterwards': True, 'refiner_checkpoint': None, 'refiner_switch_at': None, 'disable_extra_networks': False, 'comments': None, 'enable_hr': True, 'firstphase_width': 0, 'firstphase_height': 0, 'hr_scale': 1.45, 'hr_upscaler': 'ESRGAN_4x', 'hr_second_pass_steps': 20, 'hr_resize_x': 0, 'hr_resize_y': 0, 'hr_checkpoint_name': 'Dautless', 'hr_sampler_name': 'DPM++ 2M Karras', 'hr_prompt': '', 'hr_negative_prompt': '', 'sampler_index': None}

    # Call your gen_img2 function with these settings
    gen_img2(Txt2ImgRequest(**specific_args))

if __name__ == "__main__":
    main()
