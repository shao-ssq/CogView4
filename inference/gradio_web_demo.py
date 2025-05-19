import os
import re
import threading
import time
from datetime import datetime, timedelta

import gradio as gr
import random
from diffusers import CogView4Pipeline
from diffusers.models import AutoencoderKL, CogView4Transformer2DModel
import torch
from openai import OpenAI

from transformers import GlmModel
from torchao.quantization import quantize_, int8_weight_only
import gc

os.environ["OPENAI_BASE_URL"] = "https://api.openai-next.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-Jbh9xA3r9y3SCkRb2736CbEa8c3341D785A421893c3487E9"
mode = os.environ.get("MODE", "4")

total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824

print(f"\033[32mCUDA版本：{torch.version.cuda}\033[0m")
print(f"\033[32mPytorch版本：{torch.__version__}\033[0m")
print(f"\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m")
print(f"\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m")

if torch.cuda.get_device_capability()[0] >= 8:
    print(f"\033[32m支持BF16\033[0m")
    dtype = torch.bfloat16
else:
    print(f"\033[32m不支持BF16，使用FP16\033[0m")
    dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "D:\\modelscope_cache\\models\\CogView4-6B"

text_encoder = None
transformer = None
if mode in ["1", "2", "3", "4"]:
    text_encoder = GlmModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    transformer = CogView4Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    quantize_(text_encoder, int8_weight_only())
    quantize_(transformer, int8_weight_only())

pipe = CogView4Pipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=dtype,
).to(device)

if mode in ["1", "3", "2", "4"]:
    pipe.enable_model_cpu_offload()

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def convert_prompt(
        prompt: str,
        key: str,
        retry_times: int = 5,
) -> str:
    os.environ["OPENAI_API_KEY"] = key
    if not key:
        return prompt
    client = OpenAI()
    prompt = clean_string(prompt)
    for i in range(retry_times):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a bilingual image description assistant that works with an image generation bot.  You work with an assistant bot that will draw anything you say . \n    For example ,For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" or "阳光透过树林的美丽清晨" will trigger your partner bot to output an image of a forest morning, as described . \n    You will be prompted by people looking to create detailed , amazing images . The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive . \n    There are a few rules to follow : \n    - Input can be in Chinese or English. If input is in English, prompt should be written in English. If input is in Chinese, prompt should be written in Chinese.\n    - You will only ever output a single image description per user request .\n    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. \n    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\n    - Do not provide the process and explanation, just return the modified description . \n    ',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "An anime girl stands amidst a dense flower bush."',
                    },
                    {
                        "role": "assistant",
                        "content": "This image is a beautifully crafted digital illustration in an anime style. It features Suzuran, a character from the video game Arknights. She stands gracefully amidst a picturesque meadow with lush green grass and scattered wildflowers that gently sway in the breeze. Her attire includes a detailed outfit with layered ruffles and intricate fastenings, reflecting both elegance and functionality typical of the game's aesthetic. Suzuran holds a small bouquet of flowers delicately in her hands, adding to the serene atmosphere.\n\nProminent fox ears extend from her head, signifying her animal spirit form or persona within the game. The background showcases rolling hills covered in dense foliage under a brilliant blue sky dotted with fluffy white clouds. Golden leaves float whimsically through the air, enhancing the magical quality of the scene. A distant building partially obscured by vegetation hints at human presence without detracting from the natural beauty surrounding Suzuran. Overall, this artwork captures a moment of tranquility and charm, embodying the essence of the character and the enchanting world she inhabits.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "Draw a bright convertible car with a sense of artistic design."',
                    },
                    {
                        "role": "assistant",
                        "content": 'The image showcases a meticulously crafted Delahaye 135 M.S. Special Roadster from 1937 in a highly polished and realistic rendering style that highlights its luxurious design and impeccable details. The car\'s body is an elegant deep brown with a glossy finish, exuding sophistication and timeless beauty. Its aerodynamic, streamlined shape features smooth curves accentuated by chrome detailing on the fenders and running boards. The front grille has three prominent circular headlights, adding to its classic charm while the dual exhaust pipes are tastefully integrated into the rear fenders.\n\nThe open cockpit area reveals sumptuous tan leather seats, emphasizing both comfort and elegance for the driver and passenger. The vehicle’s number plate reads "MB 175". Photographed against a dark gradient background, the focus remains solely on this automotive masterpiece, allowing viewers to appreciate every intricate detail without distraction. This image captures not just a car but a piece of art, reflecting the luxury and craftsmanship emblematic of pre-war European automobiles.',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "画一个白发、神情忧郁的动漫女孩，手里拿着一支香烟"',
                    },
                    {
                        "role": "assistant",
                        "content": "这幅图像是一幅动漫风格的插画，画中描绘了一位长发飘逸的白发女孩。她神情忧郁，双眼低垂，脸上带着微微的愁容。女孩穿着浅色外套，里面搭配深色衬衫和领带，增添了她沉静却时尚的外表。她的手靠近嘴边，似乎在拿着一支香烟，画作的说明也表示她在吸烟。背景描绘了一个冬季的城市场景，地面和建筑物上覆盖着积雪。几株稀疏的植物从雪地中探出，街边停着一辆车，似乎是空荡荡的街道，增强了场景的荒凉氛围。这幅作品整体风格细致精美，典型的现代动漫插画风格，既捕捉了角色的情感表达，也呈现了周围环境的静谧氛围",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "一张红色的海报，中间写有“开门大吉”"',
                    },
                    {
                        "role": "assistant",
                        "content": "这张图片采用了海报风格，色彩鲜艳，主要以红色和金色为主，寓意吉祥如意。在画面的正中间是一块红色的大匾，上面用白色大字写着“开门大吉”，四角点缀着金色的装饰图案，显得格外喜庆。匾额上方悬挂着一对红色的灯笼，增添了节日的气氛。背景左右两侧是传统的绿色中式建筑，屋顶呈现出典型的飞檐翘角设计。底部有祥云朵朵和可爱的卡通福袋，象征着好运福气满满。整张海报传达出浓厚的节日氛围。",
                    },
                    {
                        "role": "user",
                        "content": f"Create an imaginative image descriptive caption for the user input : {prompt}",
                    },
                ],
                model="glm-4-flash",
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=300,
            )
            prompt = response.choices[0].message.content
            if prompt:
                prompt = clean_string(prompt)
                break
        except Exception as e:
            pass

    return prompt


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        os.makedirs("./gradio_tmp", exist_ok=True)
        directories = ["./gradio_tmp"]
        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()


def infer(
        prompt,
        seed,
        randomize_seed,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        num_images,
        progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if randomize_seed:
        seed = random.randint(0, 65536)

    images = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=torch.Generator().manual_seed(seed),
    ).images

    return images, seed


def update_max_height(width):
    max_height = MAX_PIXELS // width
    return gr.update(maximum=max_height)


def update_max_width(height):
    max_width = MAX_PIXELS // height
    return gr.update(maximum=max_width)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=15,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                with gr.Row():
                    enhance = gr.Button("Enhance Prompt (Strongly Suggest)", scale=1)
                    run_button = gr.Button("Run", scale=1)
                with gr.Row():
                    num_images = gr.Number(
                        label="Number of Images",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=2,
                    )
                    key = gr.Textbox(
                        label="Key",
                        placeholder="Enter your key",
                        type="password",
                        max_lines=1,
                        value="sk-Jbh9xA3r9y3SCkRb2736CbEa8c3341D785A421893c3487E9",
                    )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=65536,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=32,
                        value=1024,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=3.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50,
                    )
            with gr.Column():
                result = gr.Gallery(label="Results", show_label=True)

        MAX_PIXELS = 2 ** 21
        enhance.click(convert_prompt, inputs=[prompt, key], outputs=[prompt])
        width.change(update_max_height, inputs=[width], outputs=[height])
        height.change(update_max_width, inputs=[height], outputs=[width])

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, num_images],
        outputs=[result, seed],
    )

demo.queue().launch(inbrowser=True)
