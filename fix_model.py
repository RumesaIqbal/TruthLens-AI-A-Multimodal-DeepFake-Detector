import h5py
import json

MODEL_PATH = "models/ai_vs_human_cnn.h5"

with h5py.File(MODEL_PATH, "r+") as f:
    # Read the model config
    model_config = f.attrs["model_config"]
    
    # h5py may return bytes
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")
    
    config = json.loads(model_config)
    
    # Recursively remove quantization_config from all layers
    def remove_quantization(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for v in obj.values():
                remove_quantization(v)
        elif isinstance(obj, list):
            for item in obj:
                remove_quantization(item)
    
    remove_quantization(config)
    
    # Write patched config back
    f.attrs["model_config"] = json.dumps(config)
    print("Done — quantization_config removed successfully.")