from __future__ import annotations

import csv
import json
import math
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


OBJECTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv"
PUBLISHED_IMAGES_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"
OBJECTS_CONSTITUENTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects_constituents.csv"
CONSTITUENTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/constituents.csv"

RAW_FILENAMES = {
    "objects": "objects.csv",
    "published_images": "published_images.csv",
    "objects_constituents": "objects_constituents.csv",
    "constituents": "constituents.csv",
}

MODEL_SPECS = {
    "resnet50": {
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
        "embedding_dim": 2048,
    },
    "vit_b_16": {
        "builder": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.DEFAULT,
        "embedding_dim": 768,
    },
    "efficientnet_b0": {
        "builder": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "embedding_dim": 1280,
    },
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_model_name(model_name: str) -> str:
    if model_name not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise SystemExit(f"Unsupported model_name '{model_name}'. Supported values: {supported}")
    return model_name


def download_file(url: str, path: Path) -> None:
    ensure_dir(path.parent)
    urllib.request.urlretrieve(url, path)


def download_raw_metadata(raw_dir: Path) -> dict[str, Path]:
    ensure_dir(raw_dir)
    url_map = {
        "objects": OBJECTS_URL,
        "published_images": PUBLISHED_IMAGES_URL,
        "objects_constituents": OBJECTS_CONSTITUENTS_URL,
        "constituents": CONSTITUENTS_URL,
    }
    paths: dict[str, Path] = {}
    for key, url in url_map.items():
        path = raw_dir / RAW_FILENAMES[key]
        if not path.exists():
            download_file(url, path)
        paths[key] = path
    return paths


def build_image_url(iiif_url: str, width: int = 1024) -> str:
    return f"{iiif_url}/full/!{width},{width}/0/default.jpg"


def portrait_flag(row: pd.Series) -> int:
    fields = [
        str(row.get("title", "")),
        str(row.get("classification", "")),
        str(row.get("subclassification", "")),
        str(row.get("assistivetext", "")),
    ]
    text = " ".join(fields).lower()
    keywords = ["portrait", "self-portrait", "bust", "head", "face", "profile", "sitter"]
    return int(any(keyword in text for keyword in keywords))


def painting_flag(row: pd.Series) -> int:
    classification = str(row.get("classification", "")).lower()
    subclassification = str(row.get("subclassification", "")).lower()
    medium = str(row.get("medium", "")).lower()

    if "painting" in classification or "painting" in subclassification:
        return 1

    painting_media = ["oil", "tempera", "acrylic", "watercolor", "gouache", "pastel"]
    return int(any(token in medium for token in painting_media))


def load_merged_metadata(raw_dir: Path) -> pd.DataFrame:
    paths = download_raw_metadata(raw_dir)

    objects = pd.read_csv(
        paths["objects"],
        usecols=[
            "objectid",
            "title",
            "displaydate",
            "classification",
            "subclassification",
            "medium",
        ],
    )
    published_images = pd.read_csv(
        paths["published_images"],
        usecols=[
            "uuid",
            "iiifurl",
            "iiifthumburl",
            "viewtype",
            "openaccess",
            "depictstmsobjectid",
            "assistivetext",
        ],
    )
    objects_constituents = pd.read_csv(
        paths["objects_constituents"],
        usecols=["objectid", "constituentid", "displayorder", "roletype", "role"],
    )
    constituents = pd.read_csv(
        paths["constituents"],
        usecols=["constituentid", "preferreddisplayname", "artistofngaobject", "nationality"],
    )

    image_rows = published_images[
        (published_images["openaccess"] == 1) & (published_images["viewtype"] == "primary")
    ].copy()
    image_rows = image_rows.rename(columns={"depictstmsobjectid": "objectid"})

    artist_links = objects_constituents.merge(constituents, on="constituentid", how="left")
    artist_links["artist_rank"] = artist_links["displayorder"].fillna(9999)
    artist_links["role_text"] = (
        artist_links["roletype"].fillna("").astype(str) + " " + artist_links["role"].fillna("").astype(str)
    ).str.lower()
    artist_links["artist_hint"] = (
        artist_links["artistofngaobject"].fillna(0).astype(int)
        | artist_links["role_text"].str.contains("artist|painter|draftsman|maker").astype(int)
    )
    artist_links = artist_links.sort_values(["objectid", "artist_hint", "artist_rank"], ascending=[True, False, True])
    artist_links = artist_links.drop_duplicates("objectid")
    artist_links = artist_links[["objectid", "constituentid", "preferreddisplayname", "nationality"]]
    artist_links = artist_links.rename(
        columns={
            "preferreddisplayname": "artist_name",
            "nationality": "artist_nationality",
        }
    )

    merged = image_rows.merge(objects, on="objectid", how="inner")
    merged = merged.merge(artist_links, on="objectid", how="left")
    merged["image_url"] = merged["iiifurl"].map(build_image_url)
    merged["painting_flag"] = merged.apply(painting_flag, axis=1)
    merged["portrait_flag"] = merged.apply(portrait_flag, axis=1)
    merged = merged.drop_duplicates("objectid").reset_index(drop=True)
    return merged


def prepare_similarity_metadata(
    raw_dir: Path,
    processed_dir: Path,
    portrait_only: bool = False,
    max_rows: int = 0,
) -> Path:
    ensure_dir(processed_dir)
    metadata = load_merged_metadata(raw_dir)
    metadata = metadata[metadata["painting_flag"] == 1].copy()
    if portrait_only:
        metadata = metadata[metadata["portrait_flag"] == 1].copy()
    metadata = metadata.sort_values(["portrait_flag", "objectid"], ascending=[False, True]).reset_index(drop=True)
    if max_rows:
        metadata = metadata.head(max_rows).copy()

    output_path = processed_dir / "nga_similarity_metadata.csv"
    metadata.to_csv(output_path, index=False)
    return output_path


def download_images(metadata_csv: Path, image_dir: Path, limit: int = 0) -> int:
    ensure_dir(image_dir)
    metadata = pd.read_csv(metadata_csv)
    if limit:
        metadata = metadata.head(limit)

    downloaded = 0
    for row in metadata.itertuples(index=False):
        image_path = image_dir / f"{row.objectid}.jpg"
        if image_path.exists():
            continue
        try:
            urllib.request.urlretrieve(row.image_url, image_path)
            downloaded += 1
        except Exception:
            continue
    return downloaded


class ImageEmbeddingDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, image_dir: Path, transform: transforms.Compose) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        image_path = self.image_dir / f"{row['objectid']}.jpg"
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), index


def build_transform() -> transforms.Compose:
    return build_transform_for_model("resnet50")


def build_transform_for_model(model_name: str) -> transforms.Compose:
    spec = MODEL_SPECS[validate_model_name(model_name)]
    return spec["weights"].transforms()


def build_encoder(model_name: str = "resnet50") -> tuple[nn.Module, int]:
    spec = MODEL_SPECS[validate_model_name(model_name)]
    model = spec["builder"](weights=spec["weights"])
    if model_name == "resnet50":
        backbone = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "vit_b_16":
        model.heads = nn.Identity()
        backbone = model
    elif model_name == "efficientnet_b0":
        model.classifier = nn.Identity()
        backbone = model
    else:
        raise SystemExit(f"Model implementation missing for {model_name}")
    backbone.eval()
    return backbone, int(spec["embedding_dim"])


def extract_embeddings(
    metadata_csv: Path,
    image_dir: Path,
    output_dir: Path,
    model_name: str = "resnet50",
    batch_size: int = 16,
    num_workers: int = 0,
    device: str | None = None,
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    model_name = validate_model_name(model_name)
    metadata = pd.read_csv(metadata_csv)
    available_rows = []
    for row in metadata.itertuples(index=False):
        if (image_dir / f"{row.objectid}.jpg").exists():
            available_rows.append(row._asdict())
    metadata = pd.DataFrame(available_rows)
    if metadata.empty:
        raise SystemExit("No local images found. Run download_images.py first.")

    transform = build_transform_for_model(model_name)
    dataset = ImageEmbeddingDataset(metadata, image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model, embedding_dim = build_encoder(model_name=model_name)
    model = model.to(device or default_device())
    embeddings = torch.zeros((len(metadata), embedding_dim), dtype=torch.float32)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device or default_device())
            feats = model(images).flatten(1).cpu()
            embeddings[indices] = feats

    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    metadata_out = output_dir / "index_metadata.csv"
    embed_out = output_dir / "embeddings.pt"
    info_out = output_dir / "index_info.json"
    metadata.to_csv(metadata_out, index=False)
    torch.save(embeddings, embed_out)
    info_out.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "device": device or default_device(),
                "index_rows": len(metadata),
            },
            indent=2,
        )
    )
    return metadata_out, embed_out


def load_index(output_dir: Path) -> tuple[pd.DataFrame, torch.Tensor]:
    metadata = pd.read_csv(output_dir / "index_metadata.csv")
    embeddings = torch.load(output_dir / "embeddings.pt", map_location="cpu")
    return metadata, embeddings


def load_index_info(output_dir: Path) -> dict[str, object]:
    info_path = output_dir / "index_info.json"
    if not info_path.exists():
        return {"model_name": "resnet50", "embedding_dim": 2048}
    return json.loads(info_path.read_text())


def encode_single_image(image_path: Path, model_name: str = "resnet50", device: str | None = None) -> torch.Tensor:
    transform = build_transform_for_model(model_name)
    model, _ = build_encoder(model_name=model_name)
    model = model.to(device or default_device())
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device or default_device())
    with torch.no_grad():
        embedding = model(tensor).flatten(1).cpu()
    return torch.nn.functional.normalize(embedding, dim=1)[0]


def find_similar(
    embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    top_k: int = 8,
    exclude_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = embeddings @ query_embedding
    if exclude_index is not None:
        scores[exclude_index] = -math.inf
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
    return top_indices, top_scores


def precision_recall_ap(relevant: list[int], retrieved: list[int], top_k: int) -> tuple[float, float, float, float]:
    retrieved = retrieved[:top_k]
    relevant_set = set(relevant)
    hits = [1 if item in relevant_set else 0 for item in retrieved]
    hit_count = sum(hits)
    precision = hit_count / max(len(retrieved), 1)
    recall = hit_count / max(len(relevant_set), 1)

    precisions = []
    first_hit_rank = 0.0
    running_hits = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precisions.append(running_hits / rank)
            if first_hit_rank == 0.0:
                first_hit_rank = 1.0 / rank
    ap = sum(precisions) / max(len(relevant_set), 1)
    return precision, recall, ap, first_hit_rank


def evaluate_retrieval(
    metadata: pd.DataFrame,
    embeddings: torch.Tensor,
    label_column: str,
    top_k: int,
) -> dict[str, float]:
    if label_column not in metadata.columns:
        raise SystemExit(f"Unknown label column: {label_column}")

    precision_scores = []
    recall_scores = []
    ap_scores = []
    rr_scores = []

    labels = metadata[label_column].fillna("").astype(str).tolist()
    for query_index, query_label in enumerate(labels):
        if not query_label:
            continue
        relevant = [idx for idx, value in enumerate(labels) if value == query_label and idx != query_index]
        if not relevant:
            continue
        retrieved_indices, _ = find_similar(embeddings, embeddings[query_index], top_k=top_k + 1, exclude_index=query_index)
        retrieved = retrieved_indices.tolist()
        precision, recall, ap, rr = precision_recall_ap(relevant, retrieved, top_k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        ap_scores.append(ap)
        rr_scores.append(rr)

    return {
        f"precision@{top_k}": sum(precision_scores) / max(len(precision_scores), 1),
        f"recall@{top_k}": sum(recall_scores) / max(len(recall_scores), 1),
        f"map@{top_k}": sum(ap_scores) / max(len(ap_scores), 1),
        "mrr": sum(rr_scores) / max(len(rr_scores), 1),
        "queries_evaluated": float(len(precision_scores)),
    }


def write_retrieval_results(
    output_csv: Path,
    query_object_id: int,
    neighbors: Iterable[dict[str, object]],
) -> None:
    ensure_dir(output_csv.parent)
    fieldnames = ["query_objectid", "rank", "objectid", "score", "title", "artist_name", "classification"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in neighbors:
            writer.writerow({"query_objectid": query_object_id, **row})
