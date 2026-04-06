import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class UnifiedDRDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.samples = []
        self.transform = transform

        # Example folder structure assumption:
        # root/
        #   eyepacs/
        #   aptos/
        #   messidor/

        sources = ["eyepacs", "aptos", "messidor"]

        for source_id, source in enumerate(sources):

            img_dir = os.path.join(root_dir, source, "images")
            csv_path = os.path.join(root_dir, source, "labels.csv")

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                self.samples.append({
                    "img_path": os.path.join(img_dir, row["image"]),
                    "label": int(row["label"]),
                    "source": source_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")
        label = sample["label"]
        source = sample["source"]

        if self.transform:
            img = self.transform(img)

        return img, label, source
