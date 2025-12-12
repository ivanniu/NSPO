from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig

def merge_fsdp_model(
    local_dir: str,
    target_dir: str,
    hf_model_config_path: str,
    operation: str = "merge",
    backend: str = "fsdp"
):
    """
    Merge FSDP-sharded model checkpoints and save as a consolidated Hugging Face model.

    Args:
        local_dir (str): Path to the local FSDP checkpoint directory (with .bin shards).
        target_dir (str): Path where the merged HF model should be saved.
        hf_model_config_path (str): Path to the Hugging Face config directory (config.json, tokenizer, etc.).
        operation (str): Operation type, default is "merge".
        backend (str): Backend for merging, default is "fsdp".
    """
    config = ModelMergerConfig(
        operation=operation,
        backend=backend,
        local_dir=local_dir,
        target_dir=target_dir,
        hf_model_config_path=hf_model_config_path,
    )

    merger = FSDPModelMerger(config=config)
    merger.merge_and_save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge FSDP model checkpoints into a HF format.")
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--hf_model_config_path", type=str, required=True)
    
    args = parser.parse_args()

    merge_fsdp_model(
        local_dir=args.local_dir,
        target_dir=args.target_dir,
        hf_model_config_path=args.hf_model_config_path,
    )