import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ..config.settings import PipelineSettings

# Cache for audio models
_ASR_MODEL_CACHE = {}

def load_asr_pipeline(settings: PipelineSettings):
    """Load Whisper ASR model on GPU."""
    model_id = settings.asr_model_id
    
    if model_id in _ASR_MODEL_CACHE:
        return _ASR_MODEL_CACHE[model_id]
    
    # Clear other models before loading
    unload_asr_pipeline()
    
    print(f"Loading ASR model {model_id} onto GPU...")
    
    torch_dtype = torch.float16
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
        device=0, # Force GPU
    )
    
    _ASR_MODEL_CACHE[model_id] = pipe
    return pipe

def unload_asr_pipeline():
    """Clear ASR model from GPU memory."""
    global _ASR_MODEL_CACHE
    if _ASR_MODEL_CACHE:
        print("Unloading ASR model from GPU...")
        _ASR_MODEL_CACHE.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
