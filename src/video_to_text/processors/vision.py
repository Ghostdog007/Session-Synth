import base64
import io
import os
from pathlib import Path

import cv2
import requests
from PIL import Image

from ..config.settings import PipelineSettings
from ..types import VideoChunk, VisualNote


def extract_frames(video_path: Path, timestamps: list[float], max_size: int = 384) -> list[Image.Image]:
    """Extract frames from a video at specific timestamps, resized for VRAM efficiency."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames = []
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # Resize to fit VRAM — critical for 8GB GPUs
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            frames.append(img)
    
    cap.release()
    return frames


from ..models.loader import load_vision_model

def get_multimodal_description(frames: list[Image.Image], settings: PipelineSettings) -> str:
    """Run local GPU inference for multimodal description."""
    import torch
    try:
        engine, *engine_data = load_vision_model(settings)
        
        if engine == "transformers":
            model, processor = engine_data
            
            # Prepare the prompt for Qwen2.5-VL
            content = []
            for img in frames:
                content.append({"type": "image", "image": img})
            
            content.append({
                "type": "text", 
                "text": "Describe what is happening in this sequence of video frames. Focus on: what is shown on screen, any text visible, and the main activity. Be concise."
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            
            # Generate with memory-efficient settings
            with torch.amp.autocast('cuda'):
                generated_ids = model.generate(**inputs, max_new_tokens=200)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Free intermediate tensors
            del inputs, generated_ids, generated_ids_trimmed
            from ..models.device import empty_cache
            empty_cache()
            
            return output_text[0]
            
        elif engine == "llama_cpp":
            model = engine_data[0]
            
            def image_to_base64_data_uri(img):
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_str}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is happening in this sequence of video frames. Focus on: what is shown on screen, any text visible, and the main activity. Be concise."}
                    ]
                }
            ]
            
            for img in frames:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_data_uri(img)}
                })
            
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=200
            )
            return response["choices"][0]["message"]["content"]
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            if len(frames) > 1:
                # Retry with just the middle frame
                print(f"  OOM with {len(frames)} frames, retrying with 1 frame...")
                return get_multimodal_description([frames[len(frames)//2]], settings)
            return f"[VRAM Error: Frame too large even at 1 frame. Try closing other GPU apps (e.g. Brave browser).]"
        raise
    except Exception as e:
        import traceback
        return f"[Local Inference Error: {e}]\n{traceback.format_exc()}"


def sample_frame_timestamps(chunk: VideoChunk, settings: PipelineSettings) -> list[float]:
    end_seconds = chunk.end_seconds if chunk.end_seconds is not None else chunk.start_seconds + settings.chunk_seconds
    duration = max(0.0, end_seconds - chunk.start_seconds)
    if duration == 0:
        return [chunk.start_seconds]

    frame_count = min(settings.max_frames_per_chunk, max(1, int(duration * settings.sample_fps)))
    if frame_count == 1:
        return [chunk.start_seconds + duration / 2.0]

    step = duration / float(frame_count - 1)
    return [chunk.start_seconds + step * index for index in range(frame_count)]


def get_real_visual_notes(chunk: VideoChunk, settings: PipelineSettings) -> list[VisualNote]:
    frame_timestamps = sample_frame_timestamps(chunk, settings)
    model_id = settings.video_model.resolve_model_id()
    
    try:
        frames = extract_frames(chunk.source_path, frame_timestamps)
        if not frames:
            text = "[Warning: No frames could be extracted from this chunk]"
        else:
            text = get_multimodal_description(frames, settings)
    except Exception as e:
        text = f"[Visual processing error: {e}]"

    return [
        VisualNote(
            chunk_index=chunk.index,
            model_id=model_id,
            text=text,
            frame_timestamps=frame_timestamps,
        )
    ]


def placeholder_visual_notes(chunk: VideoChunk, settings: PipelineSettings) -> list[VisualNote]:
    """Maintained for backward compatibility or as a toggle."""
    return get_real_visual_notes(chunk, settings)
