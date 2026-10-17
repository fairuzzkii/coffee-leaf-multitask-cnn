"""Script untuk menggabungkan dataset JMUBEN, JMUBEN V2, dan BRACOL menjadi 5 kelas terpadu.

Kelas:
0: Healthy
1: Miner
2: Leaf rust
3: Phoma
4: Cercospora
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def prepare_combined_dataset(
    workspace_root="c:/Users/fairu/Documents/fairuz/Penelitian Pak ranggi - Skripsi",
    output_csv=None,
    sample_per_class=None,
):
    workspace_root = os.path.abspath(workspace_root)
    if output_csv is None:
        output_csv = os.path.join(
            workspace_root,
            "coffee-leaf-multitask-cnn",
            "classification",
            "dataset",
            "dataset_combined.csv",
        )

    print("Mengumpulkan data dari JMUBEN, JMUBEN V2, dan BRACOL...")

    records = []

    # 1. JMUBEN V2: Healthy (0) & Miner (1)
    jmuben_v2_healthy = os.path.join(workspace_root, "JMUBEN V2", "Healthy")
    if os.path.exists(jmuben_v2_healthy):
        for img in glob.glob(os.path.join(jmuben_v2_healthy, "*.*")):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.relpath(img, workspace_root),
                    "filename": os.path.basename(img),
                    "class_id": 0,
                    "class_name": "Healthy",
                    "source": "JMUBEN_V2",
                    "severity": 0  # Daun sehat keparahan = 0
                })

    jmuben_v2_miner = os.path.join(workspace_root, "JMUBEN V2", "Miner")
    if os.path.exists(jmuben_v2_miner):
        for img in glob.glob(os.path.join(jmuben_v2_miner, "*.*")):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.relpath(img, workspace_root),
                    "filename": os.path.basename(img),
                    "class_id": 1,
                    "class_name": "Miner",
                    "source": "JMUBEN_V2",
                    "severity": -1  # Severity tidak diketahui
                })

    # 2. JMUBEN: Leaf rust (2), Phoma (3), Cerscospora (4)
    jmuben_rust = os.path.join(workspace_root, "JMUBEN", "Leaf rust")
    if os.path.exists(jmuben_rust):
        for img in glob.glob(os.path.join(jmuben_rust, "*.*")):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.relpath(img, workspace_root),
                    "filename": os.path.basename(img),
                    "class_id": 2,
                    "class_name": "Leaf rust",
                    "source": "JMUBEN",
                    "severity": -1
                })

    jmuben_phoma = os.path.join(workspace_root, "JMUBEN", "Phoma")
    if os.path.exists(jmuben_phoma):
        for img in glob.glob(os.path.join(jmuben_phoma, "*.*")):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.relpath(img, workspace_root),
                    "filename": os.path.basename(img),
                    "class_id": 3,
                    "class_name": "Phoma",
                    "source": "JMUBEN",
                    "severity": -1
                })

    jmuben_cercospora = os.path.join(workspace_root, "JMUBEN", "Cerscospora")
    if os.path.exists(jmuben_cercospora):
        for img in glob.glob(os.path.join(jmuben_cercospora, "*.*")):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "filepath": os.path.relpath(img, workspace_root),
                    "filename": os.path.basename(img),
                    "class_id": 4,
                    "class_name": "Cercospora",
                    "source": "JMUBEN",
                    "severity": -1
                })

    # 3. BRACOL Dataset
    bracol_csv = os.path.join(
        workspace_root,
        "BRACOL_coffee_leaf_ images_datasets",
        "coffee-datasets",
        "coffee-datasets",
        "leaf",
        "dataset.csv",
    )
    bracol_img_dir = os.path.join(
        workspace_root,
        "BRACOL_coffee_leaf_ images_datasets",
        "coffee-datasets",
        "coffee-datasets",
        "leaf",
        "images",
    )

    class_names_map = {
        0: "Healthy",
        1: "Miner",
        2: "Leaf rust",
        3: "Phoma",
        4: "Cercospora"
    }

    if os.path.exists(bracol_csv) and os.path.exists(bracol_img_dir):
        df_bracol = pd.read_csv(bracol_csv)
        for _, row in df_bracol.iterrows():
            stress = int(row["predominant_stress"])
            if stress in class_names_map:
                img_name = f"{int(row['id'])}.jpg"
                img_path = os.path.join(bracol_img_dir, img_name)
                if os.path.exists(img_path):
                    records.append({
                        "filepath": os.path.relpath(img_path, workspace_root),
                        "filename": img_name,
                        "class_id": stress,
                        "class_name": class_names_map[stress],
                        "source": "BRACOL",
                        "severity": int(row["severity"]) if "severity" in row else -1
                    })

    df = pd.DataFrame(records)
    print(f"\nTotal data terkumpul: {len(df)} gambar")
    print("Distribusi per kelas dan sumber:")
    print(df.groupby(["class_name", "source"]).size().unstack(fill_value=0))

    # Optional: undersampling jika user mau subset seimbang
    if sample_per_class:
        print(f"\nMengambil sampel seimbang {sample_per_class} per kelas...")
        sampled_dfs = []
        for c in range(5):
            df_c = df[df["class_id"] == c]
            n = min(len(df_c), sample_per_class)
            sampled_dfs.append(df_c.sample(n=n, random_state=42))
        df = pd.concat(sampled_dfs).reset_index(drop=True)

    # Buat Stratified 5-Fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=150)
    df["fold"] = 1
    for fold_num, (_, val_idx) in enumerate(skf.split(df, df["class_id"]), start=1):
        df.loc[val_idx, "fold"] = fold_num

    # Standar split: 70% Train, 15% Val, 15% Test
    df["split"] = "train"
    # Fold 1 -> Test (20%), Fold 2 -> Val (20%), Fold 3,4,5 -> Train (60%) atau 70/15/15
    np.random.seed(150)
    rand_vals = np.random.rand(len(df))
    df.loc[rand_vals < 0.70, "split"] = "train"
    df.loc[(rand_vals >= 0.70) & (rand_vals < 0.85), "split"] = "val"
    df.loc[rand_vals >= 0.85, "split"] = "test"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nBerhasil menyimpan dataset gabungan ke: {output_csv}")
    print(f"Ringkasan split: {df['split'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gabungkan dataset JMUBEN, JMUBEN V2, dan BRACOL"
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=None,
        help="Jumlah sampel per kelas (misal 1000 atau 2000 untuk dataset seimbang)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path output file CSV hasil gabungan",
    )
    args = parser.parse_args()

    prepare_combined_dataset(
        sample_per_class=args.sample_per_class, output_csv=args.output_csv
    )

