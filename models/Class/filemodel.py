from pathlib import Path
import os
import torch
from torch import nn

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

def checkfile_fn(name):
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    return print(f"Model file exists at: {os.path.abspath(MODEL_SAVE_PATH)}" if os.path.isfile(MODEL_SAVE_PATH) 
    else f"Model file does not exist at: {os.path.abspath(MODEL_SAVE_PATH)}")

def save_fn(name, model):
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    return print(f"Saving model to: {MODEL_SAVE_PATH}") and torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) 

def load_fn(name, model):
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    return print(f"load model from: {MODEL_SAVE_PATH} completed.") and model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))