import csv
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

PROTOCOL_NAME = "CUB-10/10 Easy-OSR"
PROTOCOL_NOTE = "Custom CUB-10/10 Easy-OSR protocol. This is not an official SSB benchmark split."
REQUIRED_FILES = [
    "images", "images.txt", "image_class_labels.txt", "train_test_split.txt",
    "classes.txt", "attributes", "bounding_boxes.txt"
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cub_root(data_root="./data"):
    return os.path.join(data_root, "CUB_200_2011")


def validate_cub_root(root, require_attributes=True):
    missing = [p for p in REQUIRED_FILES if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise FileNotFoundError("Incomplete CUB_200_2011 directory. Missing: {}. Expected root: {}".format(missing, root))
    attr = os.path.join(root, "attributes", "image_attribute_labels.txt")
    if require_attributes and not os.path.exists(attr):
        raise FileNotFoundError("Missing CUB image-level attributes: {}".format(attr))
    meta = load_cub_metadata(root)
    if len(meta["class_names"]) != 200:
        raise ValueError("CUB class count mismatch: expected 200, got {}".format(len(meta["class_names"])))
    if len(meta["images"]) != 11788:
        raise ValueError("CUB image count mismatch: expected 11788, got {}".format(len(meta["images"])))
    return True


def load_cub_metadata(root):
    class_names = {}
    with open(os.path.join(root, "classes.txt"), "r", encoding="utf-8") as f:
        for line in f:
            cid, name = line.strip().split(maxsplit=1)
            class_names[int(cid) - 1] = name

    image_paths = {}
    with open(os.path.join(root, "images.txt"), "r", encoding="utf-8") as f:
        for line in f:
            iid, path = line.strip().split(maxsplit=1)
            image_paths[int(iid)] = path

    image_labels = {}
    with open(os.path.join(root, "image_class_labels.txt"), "r", encoding="utf-8") as f:
        for line in f:
            iid, cid = line.strip().split()
            c0 = int(cid) - 1
            if not 0 <= c0 < 200:
                raise ValueError("CUB label out of 0..199 after subtracting 1: {}".format(c0))
            image_labels[int(iid)] = c0

    split = {}
    with open(os.path.join(root, "train_test_split.txt"), "r", encoding="utf-8") as f:
        for line in f:
            iid, is_train = line.strip().split()
            split[int(iid)] = int(is_train)

    bboxes = {}
    bbox_path = os.path.join(root, "bounding_boxes.txt")
    if os.path.exists(bbox_path):
        with open(bbox_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes[int(parts[0])] = tuple(float(x) for x in parts[1:])

    images = []
    for iid in sorted(image_paths):
        images.append({
            "image_id": iid,
            "relative_path": image_paths[iid],
            "path": os.path.join(root, "images", image_paths[iid]),
            "class_id": image_labels[iid],
            "is_train": split[iid],
            "bbox": bboxes.get(iid),
        })
    return {"class_names": class_names, "images": images}


def load_class_attributes(root):
    """Aggregate image-level CUB attributes into class vectors.

    Official image_attribute_labels.txt columns are:
    image_id attribute_id is_present certainty_id time.
    We use is_present * certainty_id / 4 as a confidence-weighted presence score.
    """
    attr_path = os.path.join(root, "attributes", "image_attribute_labels.txt")
    if not os.path.exists(attr_path):
        raise FileNotFoundError(attr_path)
    meta = load_cub_metadata(root)
    image_to_class = {m["image_id"]: m["class_id"] for m in meta["images"]}
    sums = None
    counts = None
    with open(attr_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError("Unexpected CUB attribute row: {}".format(line[:80]))
            iid = int(parts[0]); aid = int(parts[1]) - 1
            present = float(parts[2]); certainty = float(parts[3])
            if sums is None:
                # CUB has 312 attributes; allocate dynamically in case file differs.
                sums = np.zeros((200, max(312, aid + 1)), dtype=np.float64)
                counts = np.zeros((200, max(312, aid + 1)), dtype=np.float64)
            if aid >= sums.shape[1]:
                pad = aid + 1 - sums.shape[1]
                sums = np.pad(sums, ((0, 0), (0, pad)))
                counts = np.pad(counts, ((0, 0), (0, pad)))
            c = image_to_class[iid]
            sums[c, aid] += present * (certainty / 4.0)
            counts[c, aid] += 1.0
    if sums is None:
        raise ValueError("No attributes parsed from {}".format(attr_path))
    attrs = sums / np.maximum(counts, 1.0)
    norms = np.linalg.norm(attrs, axis=1, keepdims=True)
    attrs = attrs / np.maximum(norms, 1e-12)
    return attrs, attr_path


def cub_train_transform(image_size=224, resize_size=256):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def cub_test_transform(image_size=224, resize_size=256):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CUB200Dataset(Dataset):
    def __init__(self, root, classes, split, transform=None, class_to_mapped=None, unknown=False, use_bbox=False):
        self.root = root
        self.classes = list(classes)
        self.split = split
        self.transform = transform
        self.class_to_mapped = class_to_mapped or {c: i for i, c in enumerate(self.classes)}
        self.unknown = unknown
        self.use_bbox = use_bbox
        meta = load_cub_metadata(root)
        want_train = 1 if split == "train" else 0
        self.samples = []
        for item in meta["images"]:
            if item["class_id"] not in self.classes:
                continue
            if split in ["train", "test"] and item["is_train"] != want_train:
                continue
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        img = Image.open(item["path"]).convert("RGB")
        if self.use_bbox and item.get("bbox") is not None:
            x, y, w, h = item["bbox"]
            img = img.crop((x, y, x + w, y + h))
        if self.transform is not None:
            img = self.transform(img)
        original = int(item["class_id"])
        mapped = -1 if self.unknown else int(self.class_to_mapped[original])
        return img, mapped, original, item["path"]


def stratified_train_val_indices(dataset, val_ratio=0.2, seed=123):
    by_class = {}
    for i, item in enumerate(dataset.samples):
        by_class.setdefault(item["class_id"], []).append(i)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in sorted(by_class):
        idx = np.asarray(by_class[c])
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return train_idx, val_idx


class CUB200_OSR:
    def __init__(self, split_json, data_root="./data", batch_size=128, num_workers=0,
                 image_size=224, resize_size=256, val_ratio=0.2, data_split_seed=123,
                 use_bbox=False, pin_memory=False):
        self.split_json = split_json
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        self.split = split
        self.known = list(split["known_classes"])
        self.unknown = list(split["unknown_classes"])
        self.num_known = len(self.known)
        self.root = cub_root(data_root)
        self.class_to_mapped = {c: i for i, c in enumerate(self.known)}
        self.mapped_to_class = {i: c for c, i in self.class_to_mapped.items()}

        train_base = CUB200Dataset(self.root, self.known, "train", cub_train_transform(image_size, resize_size), self.class_to_mapped, False, use_bbox)
        train_idx, val_idx = stratified_train_val_indices(train_base, val_ratio, data_split_seed)
        val_base = CUB200Dataset(self.root, self.known, "train", cub_test_transform(image_size, resize_size), self.class_to_mapped, False, use_bbox)
        known_test = CUB200Dataset(self.root, self.known, "test", cub_test_transform(image_size, resize_size), self.class_to_mapped, False, use_bbox)
        unknown_test = CUB200Dataset(self.root, self.unknown, "test", cub_test_transform(image_size, resize_size), self.class_to_mapped, True, use_bbox)

        self.train_dataset = Subset(train_base, train_idx)
        self.val_dataset = Subset(val_base, val_idx)
        self.test_dataset = known_test
        self.out_dataset = unknown_test
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.out_loader = DataLoader(self.out_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    def save_manifests(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
            json.dump({"class_to_mapped": self.class_to_mapped, "mapped_to_class": self.mapped_to_class}, f, indent=2)
        with open(os.path.join(out_dir, "split_classes.json"), "w", encoding="utf-8") as f:
            json.dump(self.split, f, indent=2)
        with open(os.path.join(out_dir, "sample_split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subset", "mapped_label", "original_label", "path"])
            for subset_name, ds in [("known_test", self.test_dataset), ("unknown_test", self.out_dataset)]:
                for i in range(len(ds)):
                    _img, mapped, orig, path = ds[i]
                    w.writerow([subset_name, mapped, orig, path])
