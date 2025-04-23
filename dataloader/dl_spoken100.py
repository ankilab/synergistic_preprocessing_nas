import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

class Spoken100DataLoader(Dataset):
    def __init__(self, path, subset, transform) -> None:
        super().__init__()
        np.random.seed(42)
        self.transform = transform
        self.path = path
        self.languages = ["english", "german", "french", "mandarin"]
        self.metadata = self._get_df_metadata()
        self.speakers = self.metadata["speaker"].unique()  # we have 32 speaker in total. We will use 28 for training, 2 for validation and 2 for testing

        if subset == "training":
            self.metadata = self.metadata[self.metadata["speaker"].isin(self.speakers[:28])]
        elif subset == "validation":
            self.metadata = self.metadata[self.metadata["speaker"].isin(self.speakers[28:30])]
        elif subset == "testing":
            self.metadata = self.metadata[self.metadata["speaker"].isin(self.speakers[30:])]
        else:
            raise ValueError("subset should be either 'training', 'validation' or 'testing'")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        file_path = self.metadata.iloc[idx]["file_path"]
        waveform, sample_rate = torchaudio.load(file_path)

        waveform = self.normalize(waveform, 32768.0)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, int(self.metadata.iloc[idx]["number"])
    
    def normalize(self, audio, factor):
        return audio / factor

    def _get_df_metadata(self):
        # create a df with all the file paths, language, speaker and number
        df = pd.DataFrame(columns=["file_path", "language", "speaker", "number"])
        for language in self.languages:
            path = Path(self.path) / (language + "_numbers")
            for speaker in path.iterdir():
                if not speaker.is_dir():
                    continue
                for file in speaker.iterdir():
                        df = pd.concat([df, pd.DataFrame({"file_path": [file], "language": [language], "speaker": [speaker.stem], "number": [file.stem]})])

        return df
    
    def get_class_weights(self):
        """ Not needed for this dataset as we have equal number of samples for each class """
        return [1.0] * len(self.metadata["number"].unique())