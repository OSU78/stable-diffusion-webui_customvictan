from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
import pytz
import random
# from magicGPTS import GPTDetailsScraper
import pytz
import boto3
import datetime
from botocore.exceptions import NoCredentialsError
from io import BytesIO
import re
import base64
# #Prod
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
REGION_NAME= 'eu-north-1'
BUCKET_NAME = 'gptstoresbucket'


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




#-----------------------------------_#








# Créer un client Amazon S3 avec l'option de configuration pour le protocole d'authentification
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION_NAME)




def upload_file_to_s3(file_path, bucket_name):
    """Envoyer un fichier sur Amazon S3"""
    file_name = generate_file_id(file_path)
    try:
        s3.upload_file(file_path, bucket_name, file_name)
        print(f"{file_name} a été uploadé avec succès sur Amazon S3")
        return file_name
    except FileNotFoundError:
        print(f"{file_path} n'a pas été trouvé")
    except NoCredentialsError:
        print("Clés d'accès invalides")

def generate_file_id(file_path):
    """Générer un id unique pour chaque fichier"""
    file_name = os.path.basename(file_path)
    now = datetime.datetime.now()
    timestamp = now.strftime("%H_%M_%S")
    return f"{os.path.splitext(file_name)[0]}_{timestamp}{os.path.splitext(file_name)[1]}"

# def make_file_public(bucket_name, file_name):
#     """Rendre un fichier public sur Amazon S3"""
#     s3.put_object_acl(Bucket=bucket_name, Key=file_name, ACL='public-read')
#     print(f"{file_name} est désormais public")

def generate_public_url(bucket_name, file_name):
    """Obtenir l'URL publique d'un fichier sur Amazon S3"""
    url = f"https://{bucket_name}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
    print(f"URL publique de {file_name}: {url}")
    return url

def upload_video(videos,bucket_name):

    print("------AMAZONE S3 BUCKET START------")
    # Envoyer le fichier ZIP sur Amazon S3
    file_name = upload_file_to_s3(videos, bucket_name)
    print("------S3 BUCKET SUCCESS------")
    # Rendre le fichier public
    #make_file_public(bucket_name, file_name)

    # Supprimer le fichier ZIP temporaire
    print("------REMOVE FILE------")
    os.remove(videos)
    

    # Générer et retourner l'URL publique
    return generate_public_url(bucket_name, file_name) 




#crée une fonction upload_logo qui prend en paramètre une image en base64 la convertie en image et l'upload sur le bucket s3
def upload_logo(logo,name="temp_logo",bucket_name=BUCKET_NAME,video_id=0):
    # Convertir l'image en base64 en image
    #logo = re.sub('^data:image/.+;base64,', '', logo)
    image_data = BytesIO(base64.b64decode(logo))

    # Créer un chemin temporaire pour le fichier
    temp_file_path = name+".png"  # ou un chemin plus unique

    # Écrire les données de l'image dans un fichier temporaire
    with open(temp_file_path, 'wb') as file:
        file.write(image_data.getbuffer())

    # Envoyer le fichier sur Amazon S3
    file_name = upload_file_to_s3(temp_file_path, bucket_name)

    # Supprimer le fichier temporaire après l'envoi
    os.remove(temp_file_path)
    print("------S3 BUCKET SUCCESS------")
    
    # Retourner l'URL de l'image
    return generate_public_url(bucket_name, file_name)












#-------------------------------------------------------------------------------------_#

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
        p.outpath_samples = os.getcwd()
        try:
            processed = process_images(p)
        finally:
            print("OK nice")

    # Conversion des images en base64 si nécessaire
    print(processed.images)
    print("------------")
    b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    #upload_logo(b64images[0])
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
    #print(args_dict)
    
    from modules import shared_init
    shared_init.initialize()
    startup_timer.record("initialize shared")
    
    # Extraire uniquement les clés spécifiques
    keys_to_extract = ['prompt', 'negative_prompt', 'styles', 'seed', 'subseed', 
                       'subseed_strength', 'seed_resize_from_h', 'seed_resize_from_w', 
                       'sampler_name', 'batch_size', 'n_iter', 'steps', 'cfg_scale', 
                       'width', 'height', 'restore_faces', 'tiling', 'do_not_save_samples', 
                       'do_not_save_grid', 'eta', 'denoising_strength', 's_min_uncond', 
                       's_churn', 's_tmax', 's_tmin', 's_noise', 'override_settings', 
                       'override_settings_restore_afterwards', 'refiner_checkpoint', 
                       'refiner_switch_at', 'disable_extra_networks', 'comments', 
                       'enable_hr', 'firstphase_width', 'firstphase_height', 'hr_scale', 
                       'hr_upscaler', 'hr_second_pass_steps', 'hr_resize_x', 'hr_resize_y', 
                       'hr_checkpoint_name', 'hr_sampler_name', 'hr_prompt', 
                       'hr_negative_prompt', 'sampler_index']
    
    specific_args = {key: args_dict.get(key, None) for key in keys_to_extract}
    print(specific_args)
    # Création du dictionnaire avec les valeurs par défaut des paramètres
    gen_image_settings = {'prompt': 'realistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))', 'negative_prompt': 'bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,Duplicate , two people,text ,signature,watermark', 'styles': None, 'seed': -1, 'subseed': -1, 'subseed_strength': 0, 'seed_resize_from_h': -1, 'seed_resize_from_w': -1, 'sampler_name': 'DPM++ 2M Karras', 'batch_size': 1, 'n_iter': 1, 'steps': 25, 'cfg_scale': 7.5, 'width': 512, 'height': 768, 'restore_faces': None, 'tiling': None, 'do_not_save_samples': True, 'do_not_save_grid': True, 'eta': None, 'denoising_strength': 0.2, 's_min_uncond': None, 's_churn': None, 's_tmax': None, 's_tmin': None, 's_noise': None, 'override_settings': {'sd_model_checkpoint': 'Dautless'}, 'override_settings_restore_afterwards': True, 'refiner_checkpoint': None, 'refiner_switch_at': None, 'disable_extra_networks': False, 'comments': None, 'enable_hr': True, 'firstphase_width': 0, 'firstphase_height': 0, 'hr_scale': 1.45, 'hr_upscaler': 'ESRGAN_4x', 'hr_second_pass_steps': 20, 'hr_resize_x': 0, 'hr_resize_y': 0, 'hr_checkpoint_name': 'Dautless', 'hr_sampler_name': 'DPM++ 2M Karras', 'hr_prompt': '', 'hr_negative_prompt': '', 'sampler_index': None}

    # Call your gen_img2 function with these settings
    gen_img2(Txt2ImgRequest(**specific_args))

if __name__ == "__main__":
    main()
