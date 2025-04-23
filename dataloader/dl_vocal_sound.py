import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json



class VocalSoundDataLoader(Dataset):
    def __init__(self, path, subset, transform) -> None:
        self.transform = transform
        self.path = Path(path)
        
        if subset == "training":
            data_json_file = self.path / "datafiles/tr.json"
        elif subset == "validation":
            data_json_file = self.path / "datafiles/val.json"
        elif subset == "test":
            data_json_file = self.path / "datafiles/te.json"
            
        with open(data_json_file, "r") as f:
            data = json.load(f)
            
        self.data = data["data"]
        
        # load class labels from "class_labels_indicies_vs.csv"
        class_labels_df = pd.read_csv(self.path / "class_labels_indices_vs.csv")
        self.class_labels = {row['display_name']: row['index'] for _, row in class_labels_df.iterrows()}

        # Keep the mid information
        self.class_mids = {row['mid']: row['display_name'] for _, row in class_labels_df.iterrows()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data[idx]["wav"]
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform - waveform.mean()
        
        # Take only 5 seconds of the waveform (truncate if longer or pad with zeros if shorter)
        if waveform.shape[1] > 5 * sr:
            waveform = waveform[:, :5 * sr]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, 5 * sr - waveform.shape[1]))
        
        waveform = waveform[0]
        waveform = waveform[None, ...]
        label_mid = self.data[idx]["labels"]
        label_str = self.class_mids.get(label_mid, -1)
        label = self.class_labels.get(label_str, -1)
        
        if label == -1:
            raise ValueError(f"Label {label_str} not found in class_labels_indices_vs.csv")
        
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
    
    def get_class_weights(self):
        """ Not needed for this dataset as we have equal number of samples for each class """
        unique_classes = np.unique(list(self.class_labels.values()))
        return [1.0] * len(unique_classes)
        
if __name__ == "__main__":
    i = 0
    for x, y in VocalSoundDataLoader("data/VocalSound", "training", None):
        print(x.shape)
        i += 1
        if i == 3:
            break
    
