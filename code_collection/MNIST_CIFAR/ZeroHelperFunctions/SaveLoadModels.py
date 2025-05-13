import torch

from pathlib import Path

def save_model(the_model, path="models", model_name="fullnetMNISTZero.pth"):
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)


    MODEL_NAME = model_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=the_model.state_dict(),
            f=MODEL_SAVE_PATH)
    
def load_model(skeleton_model, model_save_path, device):
    skeleton_model.load_state_dict(torch.load(f=model_save_path))
    skeleton_model = skeleton_model.to(device)
    return skeleton_model