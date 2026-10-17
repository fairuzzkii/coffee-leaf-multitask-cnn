# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:27:40 2019

@author: Guilherme
"""

import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.enums import Tasks


class CoffeeLeavesDataset(Dataset):
    """Coffee Leaves Dataset."""

    def __init__(self, csv_file, images_dir, dataset, fold=1, model_task=None, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            dataset (string) : Select the desired dataset - 'train', 'val' or 'test'
            fold (int{1,5}) : The data is changed based on the selected fold
            model_task (Tasks) : Select the model task according to the dataset.
            transforms : Image transformations
        """
        self.fold = fold
        self.data = self.split_dataset(pd.read_csv(csv_file), dataset)
        self.images_dir = images_dir
        self.task = model_task
        self.transformations = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image name from the pandas df
        img_name = os.path.join(self.images_dir, str(self.data.iloc[idx, 0]) + ".jpg")
        # Open image
        image = Image.open(img_name)

        # Apply transformations
        if self.transformations:
            image = self.transformations(image)

        # Get label of the image
        label_dis = self.data.iloc[idx, 1]
        label_dis = torch.tensor(label_dis, dtype=torch.long)

        label_sev = self.data.iloc[idx, -1]
        label_sev = torch.tensor(label_sev, dtype=torch.long)

        # Multitask
        if self.task == Tasks.MULTITASK:
            return (image, label_dis, label_sev)
        # Disease
        elif self.task == Tasks.BIOTIC_STRESS:
            return (image, label_dis)
        # Severity
        else:
            return (image, label_sev)

    def split_dataset(self, csv, dataset):
        seed = 150
        np.random.seed(seed)

        dataset_size = len(csv)
        partition_size = int(dataset_size / 5)
        indices = list(range(dataset_size))

        # Split: 70% for training, 15% for validation and 15% for test
        p1 = int(np.ceil(0.7 * dataset_size))
        p2 = int(np.ceil(0.85 * dataset_size))

        # Shuffle indices
        np.random.shuffle(indices)

        # Change folds order
        aux = (self.fold - 1) * partition_size
        indices = indices[aux:] + indices[:aux]

        # Get training and validation indices
        train_indices, val_indices, test_indices = indices[:p1], indices[p1:p2], indices[p2:]

        # Sort indices
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        if dataset == "train":
            return csv.iloc[train_indices]
        elif dataset == "val":
            return csv.iloc[val_indices]
        else:
            return csv.iloc[test_indices]


class CombinedCoffeeDataset(Dataset):
    """Dataset for combined JMUBEN + JMUBEN V2 + BRACOL coffee leaves.
    Supports both legacy 2-head outputs and modern 3-head multi-task outputs.
    """

    def __init__(
        self,
        csv_file,
        workspace_root=None,
        dataset="train",
        fold=1,
        model_task=None,
        transforms=None,
        is_3head=False,
    ):
        if workspace_root is None:
            # Root directory containing JMUBEN, JMUBEN V2, BRACOL
            self.workspace_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
        else:
            self.workspace_root = os.path.abspath(workspace_root)

        df = pd.read_csv(csv_file)
        self.task = model_task
        self.transformations = transforms
        self.is_3head = is_3head

        if "split" in df.columns:
            self.data = df[df["split"] == dataset].reset_index(drop=True)
        elif "fold" in df.columns:
            self.data = df[df["fold"] == fold].reset_index(drop=True)
        else:
            self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path = self.data.iloc[idx]["filepath"]
        full_path = os.path.join(self.workspace_root, rel_path)
        image = Image.open(full_path).convert("RGB")

        if self.transformations:
            image = self.transformations(image)

        class_id = int(self.data.iloc[idx]["class_id"])
        label_dis = torch.tensor(class_id, dtype=torch.long)

        raw_sev = (
            float(self.data.iloc[idx]["severity"])
            if "severity" in self.data.columns
            else -1.0
        )

        if self.is_3head:
            # 3-head output:
            # Severity is regression [0.0, 1.0].
            # Discrete 0..4 maps to 0.0, 0.25, 0.5, 0.75, 1.0.
            # If raw_sev < 0, mask_sev is 0.0 (masked out in loss)
            if raw_sev >= 0:
                sev_norm = min(max(raw_sev / 4.0, 0.0), 1.0)
                mask_sev = 1.0
            else:
                sev_norm = 0.0
                mask_sev = 0.0

            # Bounding box [cx, cy, w, h] normalized in [0, 1]
            if "bbox" in self.data.columns and pd.notna(self.data.iloc[idx]["bbox"]):
                try:
                    bbox_coords = [float(v) for v in str(self.data.iloc[idx]["bbox"]).split(",")]
                    bbox = torch.tensor(bbox_coords, dtype=torch.float32)
                    mask_bbox = 1.0
                except Exception:
                    bbox = torch.tensor([0.5, 0.5, 0.8, 0.8], dtype=torch.float32)
                    mask_bbox = 1.0
            else:
                bbox = torch.tensor([0.5, 0.5, 0.8, 0.8], dtype=torch.float32)
                mask_bbox = 1.0

            return (
                image,
                label_dis,
                torch.tensor([sev_norm], dtype=torch.float32),
                bbox,
                torch.tensor(mask_sev, dtype=torch.float32),
                torch.tensor(mask_bbox, dtype=torch.float32),
            )

        # Legacy 2-head behavior
        label_sev = torch.tensor(
            int(raw_sev) if raw_sev >= 0 else 0, dtype=torch.long
        )
        if self.task == Tasks.MULTITASK:
            return (image, label_dis, label_sev)
        elif self.task == Tasks.BIOTIC_STRESS:
            return (image, label_dis)
        elif self.task == Tasks.LOCALIZATION:
            return (image, torch.tensor([0.5, 0.5, 0.8, 0.8], dtype=torch.float32))
        else:
            return (image, label_sev)


