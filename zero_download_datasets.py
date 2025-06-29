import kagglehub
import shutil
import os

save_dir = os.path.join(os.getcwd(), "dataset")
os.makedirs(save_dir, exist_ok=True)

datasets = [
    "asaniczka/tmdb-movies-dataset-2023-930k-movies", # https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
    "rounakbanik/the-movies-dataset" # https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
]

for dataset in datasets:
    print(f"Step 0: Downloading {dataset}")
    download_path = kagglehub.dataset_download(dataset)

    for filename in os.listdir(download_path):
        if filename.endswith(".csv"):
            src = os.path.join(download_path, filename)
            dest = os.path.join(save_dir, filename)
            shutil.copy2(src, dest)
            print(f"Saved: {filename}")

print("Step 0: Completed.")
