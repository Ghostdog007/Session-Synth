import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from ..config.settings import PipelineSettings
from .device import get_device_type, empty_cache, get_system_ram_gb
from huggingface_hub import hf_hub_download

# Global cache to keep model in memory
_VISION_MODEL_CACHE = {}

def load_vision_model(settings: PipelineSettings):
    """Load Qwen2.5-VL model with 4-bit quantization on GPU."""
    model_id = settings.video_model.resolve_model_id()
    
    if model_id in _VISION_MODEL_CACHE:
        return _VISION_MODEL_CACHE[model_id]
    
    # Clear other models before loading
    unload_vision_model()
    
    device_type = get_device_type()
    
    if device_type == "cpu":
        from llama_cpp import Llama
        ram_gb = get_system_ram_gb()
        
        if ram_gb <= 8.5:
            print(f"Low RAM detected ({ram_gb:.1f}GB). Downgrading to 3B model for safety.")
            repo_id = "Qwen/Qwen2.5-VL-3B-Instruct-GGUF"
            filename = "qwen2.5-vl-3b-instruct-q4_k_m.gguf"
        else:
            print(f"Sufficient RAM detected ({ram_gb:.1f}GB). Using 7B model.")
            repo_id = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF"
            filename = "qwen2.5-vl-7b-instruct-q4_k_m.gguf"
            
        print(f"Downloading/Locating {filename} from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        print("Loading GGUF model via llama.cpp...")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0, # Force CPU
        )
        
        _VISION_MODEL_CACHE[model_id] = ("llama_cpp", model)
        return _VISION_MODEL_CACHE[model_id]

    print(f"Loading model {model_id} in 4-bit quantization...")
    
    # Configure 4-bit quantization to fit in 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    device_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    print(f"Loading model {model_id} onto {device_name}...")

    # Load model with specific optimizations for 8GB VRAM
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=settings.video_model.trust_remote_code,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=settings.video_model.trust_remote_code
    )
    
    _VISION_MODEL_CACHE[model_id] = ("transformers", model, processor)
    return _VISION_MODEL_CACHE[model_id]

def unload_vision_model():
    """Clear vision model from GPU memory."""
    global _VISION_MODEL_CACHE
    if _VISION_MODEL_CACHE:
        print("Unloading vision model from GPU...")
        _VISION_MODEL_CACHE.clear()
        empty_cache()
