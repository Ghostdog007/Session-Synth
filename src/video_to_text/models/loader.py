import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from ..config.settings import PipelineSettings

# Global cache to keep model in memory
_VISION_MODEL_CACHE = {}

def load_vision_model(settings: PipelineSettings):
    """Load Qwen2.5-VL model with 4-bit quantization on GPU."""
    model_id = settings.video_model.resolve_model_id()
    
    if model_id in _VISION_MODEL_CACHE:
        return _VISION_MODEL_CACHE[model_id]
    
    # Clear other models before loading
    unload_vision_model()
    
    print(f"Loading model {model_id} in 4-bit quantization...")
    
    # Configure 4-bit quantization to fit in 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model {model_id} onto {torch.cuda.get_device_name(0)}...")

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
    
    _VISION_MODEL_CACHE[model_id] = (model, processor)
    return model, processor

def unload_vision_model():
    """Clear vision model from GPU memory."""
    global _VISION_MODEL_CACHE
    if _VISION_MODEL_CACHE:
        print("Unloading vision model from GPU...")
        _VISION_MODEL_CACHE.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
