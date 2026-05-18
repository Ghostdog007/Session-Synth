import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ..config.settings import PipelineSettings
from .device import get_device_type, empty_cache

# Cache for audio models
_ASR_MODEL_CACHE = {}

def load_asr_pipeline(settings: PipelineSettings):
    """Load Whisper ASR model on GPU."""
    model_id = settings.asr_model_id
    
    if model_id in _ASR_MODEL_CACHE:
        return _ASR_MODEL_CACHE[model_id]
    
    # Clear other models before loading
    unload_asr_pipeline()
    
    device_type = get_device_type()
    
    if device_type == "cpu":
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper model {model_id} for CPU...")
        fw_model_id = model_id.replace("openai/whisper-", "") if "openai/whisper-" in model_id else model_id
        model = WhisperModel(fw_model_id, device="cpu", compute_type="int8")
        _ASR_MODEL_CACHE[model_id] = ("faster_whisper", model)
        return _ASR_MODEL_CACHE[model_id]
        
    print(f"Loading ASR model {model_id} onto {device_type}...")
    
    torch_dtype = torch.float16 if device_type != "cpu" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=0 if device_type == "cuda" else (-1 if device_type == "cpu" else device_type),
    )
    
    _ASR_MODEL_CACHE[model_id] = ("transformers", pipe)
    return _ASR_MODEL_CACHE[model_id]

def unload_asr_pipeline():
    """Clear ASR model from GPU memory."""
    global _ASR_MODEL_CACHE
    if _ASR_MODEL_CACHE:
        print("Unloading ASR model from GPU...")
        _ASR_MODEL_CACHE.clear()
        empty_cache()
