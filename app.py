import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util, sdxl_train_util
from argparse import Namespace
import train_network
import toml
import re

from scripts.caption_generation_openai import generate_caption
from scripts.training_script_generation import generate_sh_flux, generate_sh_sdxl, ScriptGenerator

MAX_IMAGES = 150
PROMPT_FOLDER = "prompts"

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    
    # tags
    tags = [ "text-to-image", base_model, "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A {base_model} LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update for the captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))

    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # if it's a caption text file skip the next bit
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # if caption_path exists, do not write
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. use the existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder

def run_captioning_openai(images, concept_sentence, additional_notes, *captions):
    print(f"run_captioning")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")

    # Initialize the list to store generated captions
    captions = list(captions)

    for i, image_path in enumerate(images):
        try:
            print(f"Processing image: {image_path}")
            if isinstance(image_path, str):  # If image is a file path
                # Generate caption using the imported function
                caption_text = generate_caption(image_path, system_prompt=additional_notes)
                
                print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
                
                # Append the concept sentence to the generated caption
                if concept_sentence:
                    caption_text = f"{concept_sentence} {caption_text}"

                # Update the captions list
                captions[i] = caption_text
            else:
                captions[i] = "Error: No caption generated"

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            captions[i] = "Error generating caption!!!"

        # Yield the updated captions list to update the UI
        yield captions

def run_captioning(images, concept_sentence, *captions):
    print(f"run_captioning")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        print(f"inputs {inputs}")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        print(f"generated_ids {generated_ids}")

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"generated_text: {generated_text}")
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(f"parsed_answer = {parsed_answer}")
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]
    if "flux" in base_model:
        # download unet
        if base_model == "flux-dev" or base_model == "flux-schnell":
            unet_folder = "models/unet"
        else:
            unet_folder = f"models/unet/{repo}"
        unet_path = os.path.join(unet_folder, model_file)
        if not os.path.exists(unet_path):
            os.makedirs(unet_folder, exist_ok=True)
            gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
            print(f"download {base_model}")
            hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

        # download vae
        vae_folder = "models/vae"
        vae_path = os.path.join(vae_folder, "ae.sft")
        if not os.path.exists(vae_path):
            os.makedirs(vae_folder, exist_ok=True)
            gr.Info(f"Downloading vae")
            print(f"downloading ae.sft...")
            hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

        # download clip
        clip_folder = "models/clip"
        clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
        if not os.path.exists(clip_l_path):
            os.makedirs(clip_folder, exist_ok=True)
            gr.Info(f"Downloading clip...")
            print(f"download clip_l.safetensors")
            hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

        # download t5xxl
        t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
        if not os.path.exists(t5xxl_path):
            print(f"download t5xxl_fp16.safetensors")
            gr.Info(f"Downloading t5xxl...")
            hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")
    elif "sdxl" in base_model:
        # download sdxl base
        checkpoint_folder = "models/checkpoints/sdxl"
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)
            gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
            print(f"download {base_model}")
            hf_hub_download(repo_id=repo, local_dir=checkpoint_folder, filename=model_file)



def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):
    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}")
    global models
    print(f"models={models}")
    if "flux" in base_model:
        global advanced_component_ids
        global original_advanced_component_values
        
        sh = generate_sh_flux(
            models=models,
            base_model=base_model,
            output_name=output_name,
            resolution=resolution,
            seed=seed,
            workers=workers,
            learning_rate=learning_rate,
            network_dim=network_dim,
            max_train_epochs=max_train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            timestep_sampling=timestep_sampling,
            guidance_scale=guidance_scale,
            vram=vram,
            sample_prompts=sample_prompts,
            sample_every_n_steps=sample_every_n_steps,
            advanced_component_ids=advanced_component_ids,
            original_advanced_component_values=original_advanced_component_values,
            advanced_components=advanced_components,
        )
    elif "sdxl" in base_model:
        sh = generate_sh_sdxl(
            models=models,
            base_model=base_model,
            output_name=output_name,
            resolution=resolution,
            seed=seed,
            workers=workers,
            learning_rate=learning_rate,
            network_dim=network_dim,
            max_train_epochs=max_train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            timestep_sampling=timestep_sampling,
            guidance_scale=guidance_scale,
            vram=vram,
            sample_prompts=sample_prompts,
            sample_every_n_steps=sample_every_n_steps,
            advanced_component_ids=advanced_component_ids,
            original_advanced_component_values=original_advanced_component_values,
            advanced_components=advanced_components,
        )
    else:
        raise ValueError(f"Invalid base model: {base_model}")

    return sh   

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def gen_toml_sdxl(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)

def update(
    base_model,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    if "sdxl" in base_model:
        toml = gen_toml_sdxl(
            dataset_folder,
            resolution,
            class_tokens,
            num_repeats
        )
    elif "flux" in base_model:
        toml = gen_toml(
            dataset_folder,
            resolution,
            class_tokens,
            num_repeats
        )
    else:
        raise ValueError(f"Invalid base model: {base_model}")
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced(base_model_name):
    print(f"Initializing advanced options for {base_model_name}")  # Better logging
    
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    if 'flux' in base_model_name:
        print("Adding Flux training arguments")  # Better logging
        flux_train_utils.add_flux_train_arguments(parser)
    elif 'sdxl' in base_model_name:
        print("Adding SDXL training arguments")  # Better logging
        sdxl_train_util.add_sdxl_training_arguments(parser)

    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
            
    # Print specific arguments we're interested in instead of entire args_info
    print(f"Number of advanced arguments: {len(args_info)}")
    if '--enable_bucket' in [opt for action in parser._actions for opt in action.option_strings]:
        print("Found enable_bucket option in arguments")

    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    
    print("Creating advanced UI components...")  # Add logging
    advanced_component_ids = []
    advanced_components = []
    component_count = 0  # Add counter
    
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            continue
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                    if action['help'] != None:
                        component.info = action['help']
                    component_count += 1
                    if component_count % 20 == 0:
                        print(f"Created {component_count} components...")
            
            if component != None:
                advanced_components.append(component)
                advanced_component_ids.append(component.elem_id)
    
    print(f"Created {len(advanced_components)} advanced UI components")  # Final count
    return advanced_components, advanced_component_ids

def get_prompt_files():
    """Retrieve a list of YAML prompt files from the specified folder."""
    return [f for f in os.listdir(PROMPT_FOLDER) if f.endswith('.yaml')]

def update_prompt(selected_name, prompts):
    """Return the prompt text for the selected name."""
    if selected_name in prompts:
        prompt_text = prompts[selected_name]
        print(f"Selected prompt: {prompt_text}")
        return prompt_text
    else:
        print(f"Selected prompt not found: {selected_name}")
        return ""

def load_prompts():
    """Load prompts from YAML files and return a dictionary of name: prompt."""
    prompts = {}
    for file_name in get_prompt_files():
        with open(os.path.join(PROMPT_FOLDER, file_name), 'r') as file:
            data = yaml.safe_load(file)
            prompts[data['name']] = data['prompt']
    return prompts


def update_advanced_settings(base_model_name, lora_name_value):
    """Update advanced settings based on selected model"""
    is_sdxl = "sdxl" in base_model_name
    
    # Update default values
    new_lr = "1e-4" if is_sdxl else "8e-4"
    new_dim = 64 if is_sdxl else 4
    
    # Get current values from advanced components
    current_values = [comp.value for comp in advanced_components]
    
    # Generate new script with updated values
    try:
        output_name = slugify(lora_name_value) if lora_name_value else ""
        sh = gen_sh(
            base_model=base_model_name,
            output_name=output_name,
            # ... rest of parameters ...
        )
    except:
        return {
            learning_rate: gr.update(value=new_lr),
            network_dim: gr.update(value=new_dim),
        }



theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")
# Load all prompts
prompts = load_prompts()
prompt_names = list(prompts.keys())
prompt_placeholder = "Select a pre-defined system prompt"
prompt_names.insert(0, prompt_placeholder)

def get_available_datasets():
    """Get list of available datasets on host"""
    datasets_path = "datasets"  # 基础数据集目录
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
        
    # 获取所有子目录
    datasets = []
    for item in os.listdir(datasets_path):
        item_path = os.path.join(datasets_path, item)
        if os.path.isdir(item_path):
            datasets.append(item_path)
    return datasets

def update_dataset_view(choice):
    if choice == "Upload New":
        return {
            images: gr.update(visible=True),
            existing_dataset: gr.update(visible=False)
        }
    else:
        return {
            images: gr.update(visible=False),
            existing_dataset: gr.update(visible=True, choices=get_available_datasets())
        }

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="Base model (edit the models.yaml file to add more to this list)", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images")
                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        with gr.Row():
                            # 添加数据集选择选项
                            dataset_choice = gr.Radio(
                                choices=["Upload New", "Use Existing"],
                                value="Upload New",
                                label="Dataset Source"
                            )
                            existing_dataset = gr.Dropdown(
                                choices=get_available_datasets(),
                                label="Select Existing Dataset",
                                visible=False
                            )
                        
                        images = gr.Files(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                            elem_id="image_upload"
                        )
                    with gr.Group(visible=False) as captioning_area:
                        with gr.Row():
                            # Existing button
                            do_captioning = gr.Button("Add AI captions with Florence-2")

                            # New button for OpenAI
                            add_openai_caption_button = gr.Button("Add AI Caption with OpenAI")
                        output_components.append(captioning_area)
                        
                        # Dropdown for selecting a system prompt
                        prompt_dropdown = gr.Dropdown(
                            label="Select System Prompt",
                            choices=prompt_names,
                            value=prompt_placeholder,
                        )
                        # Add a text input section for additional notes or modifications
                        additional_notes = gr.Textbox(
                            label="System prompt for OpenAI",
                            placeholder="Enter any additional notes or modifications here...",
                            lines=3
                        )
                        # Update the textbox when a new prompt is selected
                        prompt_dropdown.change(
                            fn=update_prompt,
                            inputs=[prompt_dropdown, gr.State(prompts)],
                            outputs=additional_notes
                        )

                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=False, elem_id="start_training")
                    output_components.append(start)
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                print(f"base_model.value={base_model.value}")
                default_lr = "1e-4" if "sdxl" in base_model.value else "8e-4"
                default_network_dim = 64 if "sdxl" in base_model.value else 4
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(
                            label="--learning_rate",
                            info="Learning Rate",
                            value=default_lr,
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=4, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(
                            value=default_network_dim,
                            minimum=4,
                            maximum=128,
                            step=4,
                            label="--network_dim",
                            info="LoRA Rank",
                            interactive=True
                        )
                    advanced_components, advanced_component_ids = init_advanced(base_model.value)
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()

    listeners = [
        base_model,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )

    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)
    add_openai_caption_button.click(
            fn=run_captioning_openai,
            inputs=[images, concept_sentence, additional_notes,  *caption_list],
            outputs=caption_list
        )
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])

    # Update the change handler to include train_script
    base_model.change(
        fn=update_advanced_settings,
        inputs=[base_model, lora_name],
        outputs=[learning_rate, network_dim, train_script]
    )

    # Add change handlers for all components that affect the script
    for component in listeners:
        if hasattr(component, 'change'):  # Check if component has change event
            component.change(
                fn=update,
                inputs=listeners,
                outputs=[train_script, train_config, dataset_folder]
            )

    # 在事件处理部分添加
    dataset_choice.change(
        fn=update_dataset_view,
        inputs=[dataset_choice],
        outputs=[images, existing_dataset]
    )

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(
        debug=True, 
        show_error=True, 
        allowed_paths=[cwd],
        server_name="0.0.0.0",  # 允许所有IP访问
        share=True  # 创建一个可分享的链接
    )
