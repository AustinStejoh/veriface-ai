import os
import urllib.request

MODEL_PATH = "models/best_model.pth"
FILE_ID = "1WHgMaLV_HW0iRUl96ll50v2rZI0rwkm7"

def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded ✅")
    else:
        print("Model already exists ✅")

download_model()