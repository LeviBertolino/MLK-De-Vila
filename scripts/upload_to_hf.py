"""
Upload MLK de Vila model to HuggingFace Hub.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login
    3. python scripts/upload_to_hf.py
"""

from huggingface_hub import HfApi, upload_folder
from pathlib import Path

# === Configuration ===
REPO_ID = "lbertolino/MLK-de-Vila-1.0-1.3B"
MODEL_DIR = Path(__file__).parent.parent / "models" / "mlk-de-vila-1.0-1.3b"

def main():
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating repo: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
    )

    # Upload the entire model folder
    print(f"Uploading model from: {MODEL_DIR}")
    print("This may take a while (~2.6GB)...")

    upload_folder(
        repo_id=REPO_ID,
        folder_path=str(MODEL_DIR),
        repo_type="model",
        commit_message="Initial release: MLK de Vila 1.0-1.3B — Financial education LLM for Brazilian periferia dialect",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
