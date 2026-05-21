"""
Prepare Mini-ImageNet from ImageNet using the Ravi & Larochelle splits.
Creates {OUTPUT_ROOT}/{train,val,test}/<wnid>/*.JPEG with 600 images per class,
resized to 84x84.

Defaults are read from the MINI_IMAGENET_ROOT and IMAGENET_TRAIN environment
variables and can be overridden via CLI flags.
"""
import argparse
import csv
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

IMG_SIZE = 84
IMGS_PER_CLASS = 600

SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/train.csv",
    "val":   "https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/val.csv",
    "test":  "https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/test.csv",
}


def get_class_ids_from_csv(url):
    """Download CSV and extract unique class IDs (WordNet IDs)."""
    tmp = "/tmp/mini_imagenet_split.csv"
    urllib.request.urlretrieve(url, tmp)
    classes = set()
    with open(tmp, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.add(row["label"])
    os.remove(tmp)
    return sorted(classes)


def process_class(wnid, src_root, split_dir):
    """Copy and resize IMGS_PER_CLASS images from ImageNet to Mini-ImageNet."""
    src_dir = os.path.join(src_root, wnid)
    dst_dir = os.path.join(split_dir, wnid)
    if not os.path.isdir(src_dir):
        print(f"  WARNING: {src_dir} not found, skipping")
        return wnid, 0

    os.makedirs(dst_dir, exist_ok=True)

    img_files = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
    )[:IMGS_PER_CLASS]

    count = 0
    for fname in img_files:
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(dst_path):
            count += 1
            continue
        try:
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dst_path)
            count += 1
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
    return wnid, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet-train",
        default=os.environ.get("IMAGENET_TRAIN", "./data/imagenet/train"),
        help="Path to ImageNet train directory (defaults to $IMAGENET_TRAIN or "
             "./data/imagenet/train).",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("MINI_IMAGENET_ROOT", "./data/mini-imagenet"),
        help="Where to write the Mini-ImageNet splits (defaults to "
             "$MINI_IMAGENET_ROOT or ./data/mini-imagenet).",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for split_name, url in SPLIT_URLS.items():
        print(f"\n=== Processing {split_name} split ===")
        class_ids = get_class_ids_from_csv(url)
        print(f"  {len(class_ids)} classes")

        split_dir = os.path.join(args.output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(process_class, wnid, args.imagenet_train, split_dir): wnid
                for wnid in class_ids
            }
            for future in as_completed(futures):
                wnid, n = future.result()
                print(f"  {wnid}: {n} images")

    print("\nDone! Dataset at:", args.output_root)


if __name__ == "__main__":
    main()
