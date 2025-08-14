# File: web_demo_gta.py

import argparse
import base64
import io
import os
import threading
import traceback
from queue import Queue

import numpy as np
import torch
from diffusers.utils import load_image
from einops import rearrange
from flask import Flask, render_template
from flask_socketio import SocketIO
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2

from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from pipeline import CausalInferenceStreamingPipeline
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper
from wan.vae.wanx_vae import get_wanx_vae_wrapper

# --- Setup Flask App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-matrix-game' #hardcoded for localhost is probs fine, probably you want smth more robust if you are deploying it
socketio = SocketIO(app, async_mode='threading')

# --- Global State ---
pipeline = None
vae = None
action_queue = Queue()
inference_thread = None
is_running = False
main_args = None

# --- GTA Action Mapping ---
CAM_VALUE = 0.1
GTA_KEY_MAP = {
    'w': torch.tensor([1, 0], dtype=torch.float32),  # Keyboard: forward
    's': torch.tensor([0, 1], dtype=torch.float32),  # Keyboard: backward
    'a': torch.tensor([0, -CAM_VALUE], dtype=torch.float32), # Mouse: turn left
    'd': torch.tensor([0, CAM_VALUE], dtype=torch.float32),  # Mouse: turn right
}

# --- Helper Functions ---
def parse_web_keys_to_gta_action(keys_pressed):
    """Translates a dict of pressed keys from the browser to GTA action tensors."""
    keyboard_cond = torch.tensor([0, 0], dtype=torch.float32)
    if keys_pressed.get('w'):
        keyboard_cond = GTA_KEY_MAP['w']
    elif keys_pressed.get('s'):
        keyboard_cond = GTA_KEY_MAP['s']

    mouse_cond = torch.tensor([0, 0], dtype=torch.float32)
    if keys_pressed.get('a'):
        mouse_cond = GTA_KEY_MAP['a']
    elif keys_pressed.get('d'):
        mouse_cond = GTA_KEY_MAP['d']

    return {
        "mouse": mouse_cond.to(device=pipeline.device, dtype=pipeline.weight_dtype),
        "keyboard": keyboard_cond.to(device=pipeline.device, dtype=pipeline.weight_dtype)
    }

def format_queue_for_display(q):
    """Converts the queue of key dicts into a human-readable list of strings."""
    display_list = []
    for keys_pressed in list(q.queue):
        active_keys = [key.upper() for key, pressed in keys_pressed.items() if pressed]
        if not active_keys:
            display_list.append("None")
        else:
            display_list.append("+".join(sorted(active_keys)))
    return display_list

def tensor_to_base64(tensor):
    """Converts a single frame tensor (C, H, W) to a base64 encoded string."""
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).byte()
    ndarr = tensor.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def inference_loop_web(initial_image_b64):
    """Main inference loop, adapted to run in a background thread and communicate via WebSockets."""
    global pipeline, vae, action_queue, is_running
    is_running = True

    # --- FIX: Added torch.no_grad() context manager to prevent memory leaks ---
    with torch.no_grad():
        try:
            socketio.emit('update_status', {'status': 'Preparing models and inputs...'})

            # --- Prep Inputs ---
            image = Image.open(io.BytesIO(base64.b64decode(initial_image_b64))).convert("RGB")
            w, h = image.size
            th, tw = 352, 640
            if h / w > th / tw: new_w, new_h = int(w), int(w * th / tw)
            else: new_h, new_w = int(h), int(h * tw / th)
            left, top = (w - new_w) // 2, (h - new_h) // 2
            image = image.crop((left, top, left + new_w, top + new_h))

            frame_process = v2.Compose([
                v2.Resize(size=(352, 640), antialias=True), v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            image_tensor = frame_process(image)[None, :, None, :, :].to(dtype=pipeline.weight_dtype, device=pipeline.device)

            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (main_args.max_num_output_frames - 1), 1, 1)
            img_cond = torch.cat([image_tensor, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            img_cond = vae.encode(img_cond, device=pipeline.device, **tiler_kwargs)
            
            mask_cond = torch.ones_like(img_cond); mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = vae.clip.encode_video(image_tensor)
            
            sampled_noise = torch.randn(
                [1, 16, main_args.max_num_output_frames, 44, 80], device=pipeline.device, dtype=pipeline.weight_dtype
            )
            
            num_frames_total_actions = (main_args.max_num_output_frames - 1) * 4 + 1
            
            conditional_dict = {
                "cond_concat": cond_concat.to(device=pipeline.device, dtype=pipeline.weight_dtype),
                "visual_context": visual_context.to(device=pipeline.device, dtype=pipeline.weight_dtype),
                'mouse_cond': torch.zeros((1, num_frames_total_actions, 2), device=pipeline.device, dtype=pipeline.weight_dtype),
                'keyboard_cond': torch.zeros((1, num_frames_total_actions, 2), device=pipeline.device, dtype=pipeline.weight_dtype)
            }
            
            p = pipeline
            p.kv_cache1 = p.kv_cache_keyboard = p.kv_cache_mouse = p.crossattn_cache = None
            
            batch_size, num_frames = sampled_noise.shape[0], sampled_noise.shape[2]
            num_blocks = num_frames // p.num_frame_per_block
            
            p._initialize_kv_cache(batch_size, sampled_noise.dtype, sampled_noise.device)
            p._initialize_kv_cache_mouse_and_keyboard(batch_size, sampled_noise.dtype, sampled_noise.device)
            p._initialize_crossattn_cache(batch_size, sampled_noise.dtype, sampled_noise.device)

            current_start_frame = 0
            vae_cache = [None] * len(ZERO_VAE_CACHE)
            
            socketio.emit('update_status', {'status': 'Ready! Use WASD to drive.'})
            from pipeline.causal_inference import cond_current

            for _ in range(num_blocks):
                if not is_running: break
                
                socketio.emit('queue_update', {'queue': format_queue_for_display(action_queue)})
                keys_pressed = action_queue.get()
                socketio.emit('queue_update', {'queue': format_queue_for_display(action_queue)})
                
                current_actions = parse_web_keys_to_gta_action(keys_pressed)
                new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, p.num_frame_per_block, replace=current_actions, mode='gta_drive')
                noisy_input = sampled_noise[:, :, current_start_frame : current_start_frame + p.num_frame_per_block]
                
                for index, current_timestep in enumerate(p.denoising_step_list):
                    timestep = torch.ones([batch_size, p.num_frame_per_block], device=sampled_noise.device, dtype=torch.int64) * current_timestep
                    _, denoised_pred = p.generator(noisy_input, new_act, timestep, p.kv_cache1, p.kv_cache_mouse, p.kv_cache_keyboard, p.crossattn_cache, current_start_frame * p.frame_seq_length)
                    
                    if index < len(p.denoising_step_list) - 1:
                        next_timestep = p.denoising_step_list[index + 1]
                        noisy_input = p.scheduler.add_noise(
                            rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                            torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                            next_timestep * torch.ones([batch_size * p.num_frame_per_block], device=sampled_noise.device, dtype=torch.long)
                        )
                        noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                
                context_timestep = torch.ones_like(timestep) * p.args.context_noise
                p.generator(denoised_pred, new_act, context_timestep, p.kv_cache1, p.kv_cache_mouse, p.kv_cache_keyboard, p.crossattn_cache, current_start_frame * p.frame_seq_length)
                
                denoised_pred_t = denoised_pred.transpose(1, 2)
                video_frames, vae_cache = p.vae_decoder(denoised_pred_t.half(), *vae_cache)
                
                socketio.emit('new_chunk_start', {})
                video_frames = video_frames.squeeze(0) 
                for frame_idx in range(video_frames.shape[0]):
                    b64_frame = tensor_to_base64(video_frames[frame_idx])
                    socketio.emit('video_frame', {'image': b64_frame})
                    socketio.sleep(1/25)
                
                current_start_frame += p.num_frame_per_block
                
                # --- FIX: Clear cache periodically to prevent fragmentation ---
                torch.cuda.empty_cache()

        except Exception as e:
            full_traceback = traceback.format_exc()
            print("An error occurred during inference:")
            print(full_traceback)
            socketio.emit('update_status', {'status': f'Error: {e}. Check server console for details.'})
        finally:
            is_running = False
            socketio.emit('update_status', {'status': 'Finished. You can start a new session.'})
            socketio.emit('queue_update', {'queue': []})

# --- Flask & SocketIO Routes ---
@app.route('/')
def index():
    return render_template('index_gta.html')

@socketio.on('start_inference')
def handle_start_inference(data):
    global inference_thread
    if is_running:
        print("Inference is already running.")
        return

    print("Starting inference...")
    image_b64 = data['image'].split(',')[1]
    
    inference_thread = socketio.start_background_task(target=inference_loop_web, initial_image_b64=image_b64)

@socketio.on('key_input')
def handle_key_input(keys_pressed):
    if is_running and not action_queue.full():
        action_queue.put(keys_pressed)
        socketio.emit('queue_update', {'queue': format_queue_for_display(action_queue)})

@socketio.on('disconnect')
def handle_disconnect():
    global is_running
    is_running = False
    print('Client disconnected. Stopping inference thread.')

def main():
    global pipeline, vae, main_args
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_gta_drive.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="Matrix-Game-2.0/gta_distilled_model/gta_keyboard2dim.safetensors")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--max_num_output_frames", type=int, default=360) 
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main_args = args
    
    set_seed(args.seed)
    print("Initializing models... This may take a moment.")
    
    config = OmegaConf.load(args.config_path)
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
    
    current_vae_decoder = VAEDecoderWrapper()
    vae_state_dict = torch.load(os.path.join(args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
    decoder_state_dict = {k: v for k, v in vae_state_dict.items() if 'decoder.' in k or 'conv2' in k}
    current_vae_decoder.load_state_dict(decoder_state_dict)
    current_vae_decoder.to(device, torch.float16).eval().requires_grad_(False)
    
    pipeline = CausalInferenceStreamingPipeline(config, generator=generator, vae_decoder=current_vae_decoder)
    
    if args.checkpoint_path:
        state_dict = load_file(args.checkpoint_path)
        pipeline.generator.load_state_dict(state_dict)
    
    pipeline.to(device=device, dtype=weight_dtype)
    pipeline.device = device
    pipeline.weight_dtype = weight_dtype
    pipeline.vae_decoder.to(torch.float16)
    
    vae = get_wanx_vae_wrapper(args.pretrained_model_path, torch.float16)
    vae.to(device, weight_dtype).eval().requires_grad_(False)
    
    print(f"\nModels loaded. Starting web server at http://{args.host}:{args.port}")
    print(f"Access the demo in your browser, e.g., http://127.0.0.1:{args.port}\n")
    
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()
