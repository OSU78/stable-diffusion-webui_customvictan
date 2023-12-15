from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:8000'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

import argparse
import json
def call_txt2img_api(**kwargs):
    print("Arguments reçus:", kwargs)
    response = call_api('sdapi/v1/txt2img', **kwargs)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Call the txt2img API with specified parameters.")
    
    # Paramètres généraux
    parser.add_argument('--prompt', type=str, help='The prompt to use for the image generation')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompts to avoid certain features')
    parser.add_argument('--seed', type=int, help='Seed for random number generation')
    parser.add_argument('--steps', type=int, help='Number of steps for the generation process')
    parser.add_argument('--width', type=int, help='Width of the generated image')
    parser.add_argument('--height', type=int, help='Height of the generated image')
    parser.add_argument('--cfg_scale', type=float, help='CFG scale')
    parser.add_argument('--sampler_name', type=str, help='Sampler name')
    parser.add_argument('--n_iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--enable_hr', type=bool, help='Enable high resolution')
    parser.add_argument('--denoising_strength', type=float, help='Denoising strength')

    # Paramètres pour la haute résolution
    parser.add_argument('--hr_scale', type=float, help='Scale for high resolution')
    parser.add_argument('--hr_upscaler', type=str, help='Upscaler for high resolution')
    parser.add_argument('--hr_second_pass_steps', type=int, help='Steps for the second pass in high resolution')
    parser.add_argument('--hr_resize_x', type=int, help='Resize value for x-axis in high resolution')
    parser.add_argument('--hr_resize_y', type=int, help='Resize value for y-axis in high resolution')
    parser.add_argument('--hr_checkpoint_name', type=str, help='Checkpoint name for high resolution')
    parser.add_argument('--hr_sampler_name', type=str, help='Sampler name for high resolution')
    parser.add_argument('--hr_prompt', type=str, help='High resolution prompt')
    parser.add_argument('--hr_negative_prompt', type=str, help='High resolution negative prompt')

    # Paramètres supplémentaires
    parser.add_argument('--override_sd_model_checkpoint', type=str, help='Override SD model checkpoint')

    # Parse the arguments
    args = parser.parse_args()

    # Convert the Namespace to a dictionary
    args_dict = vars(args)

    # Remove None values
    args_dict = {k: v for k, v in args_dict.items() if v is not None}

    # Call the API function with the parsed arguments
    call_txt2img_api(**args_dict)













#----------------------------------INFOS---------------------#

# if __name__ == '__main__':
#     payload = {
#         "prompt": "realistic oil painting, portrait of a young man, looking away from viewer, full body, red hair, detailed face, hard brush, sexy clothings, in a dark forest, night, barely lit ((head shoot, neck , face:1.4))",
#         "negative_prompt": "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs,Duplicate , two people,text ,signature,watermark",
#         "seed": -1,
#         "steps": 25,
#         "width": 512,
#         "height": 768,
#         "cfg_scale": 7.5,
#         "sampler_name": "DPM++ 2M Karras",
#         "n_iter": 1,
#         "batch_size": 1,
#         "enable_hr": True,
#         "denoising_strength": 0.2,
#         "hr_scale": 1.45,
#         "hr_upscaler": "ESRGAN_4x",
#         "hr_second_pass_steps": 20,
#         "hr_resize_x": 0,
#         "hr_resize_y": 0,
#         "hr_checkpoint_name": "Dautless",
#         "hr_sampler_name": "DPM++ 2M Karras",
#         "hr_prompt": "",
#         "hr_negative_prompt": "",
#         "override_settings": {
#             'sd_model_checkpoint': "Dautless",  # this can use to switch sd model
#         },

#         # example args for x/y/z plot
#         # "script_name": "x/y/z plot",
#         # "script_args": [
#         #     1,
#         #     "10,20",
#         #     [],
#         #     0,
#         #     "",
#         #     [],
#         #     0,
#         #     "",
#         #     [],
#         #     True,
#         #     True,
#         #     False,
#         #     False,
#         #     0,
#         #     False
#         # ],

#         # example args for Refiner and ControlNet
#         # "alwayson_scripts": {
#         #     "ControlNet": {
#         #         "args": [
#         #             {
#         #                 "batch_images": "",
#         #                 "control_mode": "Balanced",
#         #                 "enabled": True,
#         #                 "guidance_end": 1,
#         #                 "guidance_start": 0,
#         #                 "image": {
#         #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
#         #                     "mask": None  # base64, None when not need
#         #                 },
#         #                 "input_mode": "simple",
#         #                 "is_ui": True,
#         #                 "loopback": False,
#         #                 "low_vram": False,
#         #                 "model": "control_v11p_sd15_canny [d14c016b]",
#         #                 "module": "canny",
#         #                 "output_dir": "",
#         #                 "pixel_perfect": False,
#         #                 "processor_res": 512,
#         #                 "resize_mode": "Crop and Resize",
#         #                 "threshold_a": 100,
#         #                 "threshold_b": 200,
#         #                 "weight": 1
#         #             }
#         #         ]
#         #     },
#         #     "Refiner": {
#         #         "args": [
#         #             True,
#         #             "sd_xl_refiner_1.0",
#         #             0.5
#         #         ]
#         #     }
#         # },
#         # "enable_hr": True,
#         # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
#         # "hr_scale": 2,
#         # "denoising_strength": 0.5,
#         # "styles": ['style 1', 'style 2'],
#         # "override_settings": {
#         #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can use to switch sd model
#         # },
#     }
#     call_txt2img_api(**payload)

#     init_images = [
#         encode_file_to_base64(r"B:\path\to\img_1.png"),
#         # encode_file_to_base64(r"B:\path\to\img_2.png"),
#         # "https://image.can/also/be/a/http/url.png",
#     ]



#     # batch_size = 2
#     # payload = {
#     #     "prompt": "1girl, blue hair",
#     #     "seed": 1,
#     #     "steps": 20,
#     #     "width": 512,
#     #     "height": 512,
#     #     "denoising_strength": 0.5,
#     #     "n_iter": 1,
#     #     "init_images": init_images,
#     #     "batch_size": batch_size if len(init_images) == 1 else len(init_images),
#     #     # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
#     # }
#     # # if len(init_images) > 1 then batch_size should be == len(init_images)
#     # # else if len(init_images) == 1 then batch_size can be any value int >= 1
#     # call_img2img_api(**payload)

#     # there exist a useful extension that allows converting of webui calls to api payload
#     # particularly useful when you wish setup arguments of extensions and scripts
#     # https://github.com/huchenlei/sd-webui-api-payload-display
