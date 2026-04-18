import gdown
import os

MODEL_PATH = "models/best_model.pth"

def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(
                id="1WHgMaLV_HW0iRUl96ll50v2rZI0rwkm7",
                output=MODEL_PATH,
                quiet=False,
                fuzzy=True
            )
            print("Model downloaded ✅")
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    else:
        print("Model already exists ✅")

download_model()