import os
import random
from typing import List, Tuple, Optional
import cv2
from PIL import Image
from torch.utils.data import Dataset
import logging


class UltrasoundDataLoader(Dataset):
    def __init__(
        self,
        root: str = "./Ultrasound_Stone_No_Stone",
        mode: str = "train",
        transform=None,
        cache_list: bool = True,
        return_path: bool = False
    ):
        self.categories = ["Normal", "stone"]
        self.transform = transform
        self.return_path = return_path
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        data_path_mode = os.path.join(root, mode)
        assert os.path.isdir(data_path_mode), f"Invalid dataset split path: {data_path_mode}"

        cache_file = os.path.join(root, f"_{mode}_cache.pkl")
        if cache_list and os.path.exists(cache_file):
            import pickle
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.image_paths, self.labels = data["paths"], data["labels"]
        else:
            for label, category in enumerate(self.categories):
                category_path = os.path.join(data_path_mode, category)
                if not os.path.exists(category_path):
                    logging.warning(f"Missing category folder: {category_path}")
                    continue
                for file in os.listdir(category_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(category_path, file))
                        self.labels.append(label)

            if cache_list:
                import pickle
                with open(cache_file, "wb") as f:
                    pickle.dump({"paths": self.image_paths, "labels": self.labels}, f)

        # Shuffle once
        data = list(zip(self.image_paths, self.labels))
        random.shuffle(data)
        self.image_paths, self.labels = zip(*data)
        self.image_paths, self.labels = list(self.image_paths), list(self.labels)

        logging.info(f"[{mode}] Found {len(self.image_paths)} samples ({len(self.categories)} classes)")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple:
        path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image {path}: {e}")
            # # fallback: create dummy black image to keep batch consistent
            # import numpy as np
            # image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            pass
        if self.transform:
            image = self.transform(image)

        return (image, label, path) if self.return_path else (image, label)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path = "./Ultrasound_Stone_No_Stone"
    data = UltrasoundDataLoader(root=path, mode="train", transform=None)
    img, label = data.__getitem__(3)
    plt.imshow(img)
    plt.title(data.categories[label])
    plt.show()
