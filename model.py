from huggingface_hub import snapshot_download

# Replace with the folder where you want the model
local_model_path = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"

snapshot_download(
    repo_id="BAAI/bge-base-en-v1.5",
    local_dir=local_model_path,
    local_dir_use_symlinks=False
)

print("✅ Model downloaded to:", local_model_path)
