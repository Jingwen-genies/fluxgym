"""Generate training command for sd-scripts used in fluxgym backend training"""
import os
import sys
from typing import Dict, List, Any, Optional


def resolve_path(p):
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of scripts folder)
    project_root = os.path.dirname(script_dir)
    # Join with the provided path
    norm_path = os.path.normpath(os.path.join(project_root, p))
    return f"\"{norm_path}\""


class ScriptGenerator:
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.line_break = "\\"
    def _get_base_command(self, base_model: str) -> str:
        """Generate base command based on model type"""
        base_parts = [
            "accelerate launch",
            "--mixed_precision bf16",
            "--num_cpu_threads_per_process 1"
        ]
        base_cmd = f" {self.line_break}\n  ".join(base_parts) + f" {self.line_break}\n  "

        if "flux" in base_model:
            paths = self._get_model_paths(base_model)
            model_args = [
                "sd-scripts/flux_train_network.py",
                f"""--pretrained_model_name_or_path {paths['unet']} """,
                f"""--clip_l {paths['clip']} """,
                f"""--t5xxl {paths['t5xxl']} """,
                f"""--ae {paths['vae']} """
            ]
            return base_cmd + f" {self.line_break}\n  ".join(model_args)
        elif "sdxl" in base_model:
            paths = self._get_model_paths(base_model)
            model_args = [
                "sd-scripts/sdxl_train_network.py",
                f"""--pretrained_model_name_or_path {paths['pretrained']} """
            ]
            return base_cmd + f" {self.line_break}\n  ".join(model_args)
        else:
            raise ValueError(f"Unsupported model type: {base_model}")

    def _get_model_paths(self, base_model: str) -> Dict[str, str]:
        """Get model-specific paths"""
        model = self.models[base_model]
        paths = {}
        
        if "flux" in base_model:
            paths.update({
                "unet": resolve_path(f"models/unet/{model['repo']}/{model['file']}" if "/" in base_model else f"models/unet/{model['file']}"),
                "vae": resolve_path("models/vae/ae.sft"),
                "clip": resolve_path("models/clip/clip_l.safetensors"),
                "t5xxl": resolve_path("models/clip/t5xxl_fp16.safetensors")
            })
        elif "sdxl" in base_model:
            paths["pretrained"] = resolve_path(f"models/checkpoints/sdxl/{model['file']}")
            
        return paths

    def _format_basic_args(self, base_model, **kwargs) -> str:
        """Format basic training arguments with model-specific defaults"""
        output_name = kwargs['output_name']
        output_dir = resolve_path(f'outputs/{output_name}')
        data_config_path = resolve_path(f'outputs/{output_name}/dataset.toml')
        
        args = [
            f"--output_name {output_name}",
            f"--output_dir {output_dir}",
            f"--resolution {kwargs['resolution']}",
            f"--seed {kwargs['seed']}",
            f"--max_data_loader_n_workers {kwargs['workers']}",
            f"--learning_rate {kwargs['learning_rate']}",
            f"--network_dim {kwargs['network_dim']}",
            f"--max_train_epochs {kwargs['max_train_epochs']}",
            f"--save_every_n_epochs {kwargs['save_every_n_epochs']}",
            f"--dataset_config {data_config_path}",
            "--enable_bucket",
            "--min_bucket_reso 128",
            "--max_bucket_reso 2048",
        ]

        if "flux" in base_model:
            args.extend([
                f"--timestep_sampling {kwargs['timestep_sampling']}",
                f"--guidance_scale {kwargs['guidance_scale']}",
                "--save_precision bf16",
                "--network_module networks.lora_flux",
                "--cache_latents_to_disk",
                "--save_model_as safetensors",
                "--sdpa",
                "--persistent_data_loader_workers",
                "--gradient_checkpointing",
                "--cache_text_encoder_outputs",
                "--cache_text_encoder_outputs_to_disk",
                "--fp8_base",
                "--highvram",
                "--discrete_flow_shift 3.1582",
                "--model_prediction_type raw",
                "--loss_type l2",
            ])
            # Add VRAM-specific optimizer settings for Flux
            if kwargs['vram'] == "16G":
                args.extend([
                    "--optimizer_type adafactor",
                    '--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"',
                    "--lr_scheduler constant_with_warmup",
                    "--max_grad_norm 0.0"
                ])
            elif kwargs['vram'] == "12G":
                args.extend([
                    "--optimizer_type adafactor",
                    '--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"',
                    "--split_mode",
                    '--network_args "train_blocks=single"',
                    "--lr_scheduler constant_with_warmup",
                    "--max_grad_norm 0.0"
                ])
            else:  # 20G+
                args.append("--optimizer_type adamw8bit")
        elif "sdxl" in base_model:
            args.extend([
                "--save_precision bf16",
                "--network_module networks.lora",
                "--gradient_accumulation_steps 1",
                f"--text_encoder_lr {kwargs['learning_rate']}",
                f"--unet_lr {kwargs['learning_rate']}",
                "--network_alpha 128",
                "--loss_type l2",
                "--gradient_checkpointing",
                "--bucket_no_upscale",
                "--cache_latents",
                "--cache_latents_to_disk",
                "--no_half_vae",
                "--persistent_data_loader_workers",
                "--max_token_length 150",
                "--save_model_as safetensors",
                "--sdpa",
                "--save_last_n_steps_state 1",
                "--prior_loss_weight 1",
                "--max_timestep 1000",
                "--optimizer_type adafactor",
                '--optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"',
                "--lr_scheduler constant",
                "--max_grad_norm 1.0",
                "--save_state",
                "--sample_sampler euler_a",
                "--caption_extension .txt2",
                "--bucket_reso_steps 64",

            ])

        # Add sample prompts if provided
        if kwargs['sample_prompts'] and kwargs['sample_every_n_steps'] > 0:
            sample_prompts_path = resolve_path(f"outputs/{kwargs['output_name']}/sample_prompts.txt")
            args.extend([
                f"""--sample_prompts={sample_prompts_path} """,
                f"""--sample_every_n_steps="{kwargs['sample_every_n_steps']}" """
            ])

        # Join with line break
        return f" {self.line_break}\n  ".join(args)

    def _format_advanced_args(self,
                            advanced_component_ids: List[str],
                            original_advanced_component_values: Optional[List[Any]] = None,
                            advanced_components: Optional[List[Any]] = None) -> str:
        """Format advanced training arguments"""
        print(f"original_advanced_component_values = {original_advanced_component_values}")
        
        advanced_flags = []
        for i, current_value in enumerate(advanced_components):
            if original_advanced_component_values[i] != current_value:
                # dirty
                if current_value == True:
                    # Boolean flag
                    advanced_flags.append(advanced_component_ids[i])
                else:
                    # Value argument
                    advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")
                
        if len(advanced_flags) > 0:
            advanced_flags_str = f" {self.line_break}\n  ".join(advanced_flags)
            return "\n  " + advanced_flags_str
        return ""

    def generate_script(self,
                       base_model: str,
                       output_name: str,
                       resolution: int,
                       seed: int,
                       workers: int,
                       learning_rate: str,
                       network_dim: int,
                       max_train_epochs: int,
                       save_every_n_epochs: int,
                       timestep_sampling: str,
                       guidance_scale: float,
                       vram: str,
                       sample_prompts: str,
                       sample_every_n_steps: int,
                       advanced_component_ids: List[str],
                       original_advanced_component_values: Optional[List[Any]] = None,
                       advanced_components: Optional[List[Any]] = None) -> str:
        """Generate complete training script"""
        # Get base command and add model paths
        command = [self._get_base_command(base_model)]
        
        # Add basic arguments
        basic_args = self._format_basic_args(
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
            sample_every_n_steps=sample_every_n_steps
        )
        if basic_args:
            command.append(basic_args)
        
        # Add advanced arguments
        advanced_args = self._format_advanced_args(
            advanced_component_ids,
            advanced_components,
            original_advanced_component_values
        )
        if advanced_args:
            command.append(advanced_args)
        
        # Join all parts with line break
        return f" {self.line_break}\n  ".join(command)

# Create convenience functions for backward compatibility
def generate_sh_flux(*args, **kwargs):
    generator = ScriptGenerator(kwargs.pop('models'))
    return generator.generate_script(*args, **kwargs)

def generate_sh_sdxl(*args, **kwargs):
    generator = ScriptGenerator(kwargs.pop('models'))
    return generator.generate_script(*args, **kwargs)



