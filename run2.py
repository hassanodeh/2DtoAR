import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video


def main_with_args(
    config, input_path, output_path="outputs/", diffusion_steps=50,
    seed=42, scale=1.0, distance=4.5, view=6, no_rembg=False,
    export_texmap=True, save_video=False
):
    """
    Main function to generate 3D .obj files from 2D images using provided arguments.
    """
    # Set random seed
    seed_everything(seed)

    # Load configuration
    config_path = config  # Preserve the original string path
    config = OmegaConf.load(config_path)  # Load the YAML as a DictConfig
    config_name = os.path.basename(config_path).replace('.yaml', '')  # Use the string path
    model_config = config.model_config
    infer_config = config.infer_config

    IS_FLEXICUBES = config_name.startswith('instant-mesh')
    device = torch.device('cuda')

    # Load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # Load custom white-background UNet
    print('Loading custom white-background unet ...')
    unet_ckpt_path = infer_config.unet_path if os.path.exists(infer_config.unet_path) else hf_hub_download(
        repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model"
    )
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)

    # Load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    model_ckpt_path = infer_config.model_path if os.path.exists(infer_config.model_path) else hf_hub_download(
        repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model"
    )
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    # Make output directories
    image_path = os.path.join(output_path, config_name, 'images')
    mesh_path = os.path.join(output_path, config_name, 'meshes')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)

    # Process input files
    input_files = (
        [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(('.png', '.jpg', '.webp'))]
        if os.path.isdir(input_path)
        else [input_path]
    )
    print(f'Total number of input images: {len(input_files)}')

    # Initialize background removal session
    rembg_session = None if no_rembg else rembg.new_session()

    outputs = []
    for idx, image_file in enumerate(input_files):
        name = os.path.basename(image_file).split('.')[0]
        print(f'[{idx+1}/{len(input_files)}] Processing {name} ...')

        # Remove background if necessary
        input_image = Image.open(image_file)
        if not no_rembg:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)

        # Sampling
        output_image = pipeline(input_image, num_inference_steps=diffusion_steps).images[0]
        output_image.save(os.path.join(image_path, f'{name}.png'))
        print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        outputs.append({'name': name, 'images': images})

    # Free memory by deleting pipeline
    del pipeline

    # Reconstruction
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * scale).to(device)
    for idx, sample in enumerate(outputs):
        name = sample['name']
        print(f'[{idx+1}/{len(outputs)}] Generating mesh for {name} ...')

        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

        with torch.no_grad():
            planes = model.forward_planes(images, input_cameras)
            mesh_out = model.extract_mesh(planes, use_texture_map=export_texmap, **infer_config)

            # Save mesh
            mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')
            if export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_path_idx)

            print(f"Mesh saved to {mesh_path_idx}")

    print("Processing complete.")
