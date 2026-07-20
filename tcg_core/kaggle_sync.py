import os
import json

def get_kaggle_api():
    os.environ.pop("KAGGLE_API_TOKEN", None)
    os.environ.pop("KAGGLE_API_V1_TOKEN", None)
    os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    os.environ.pop("KAGGLE_DATA_PROXY_URL", None)
    os.environ["KAGGLE_USERNAME"] = "akhyarsafrudin"
    os.environ["KAGGLE_KEY"] = "03c3e536ffedc7d6153c1b3b8515242b"
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def upload_to_kaggle(save_dir, message="Update models"):
    dataset_id = "akhyarsafrudin/tcg-models"
    metadata_path = os.path.join(save_dir, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        metadata = {
            "title": dataset_id.split("/")[-1],
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    try:
        api = get_kaggle_api()
        dataset_exists = True
        try:
            api.dataset_status(dataset_id)
        except Exception as status_err:
            dataset_exists = False
        if dataset_exists:
            api.dataset_create_version(save_dir, version_notes=message, dir_mode="zip")
            print("[*] Sukses sinkronisasi ke Kaggle Dataset.")
        else:
            api.dataset_create_new(save_dir, public=False, quiet=False, convert_to_csv=False, dir_mode="zip")
            print("[*] Sukses membuat dan mengunggah ke Kaggle Dataset baru.")
    except Exception as e:
        print(f"[!] Terjadi error saat upload Kaggle: {e}")

def download_from_kaggle(save_dir):
    dataset_id = "akhyarsafrudin/tcg-models"
    try:
        api = get_kaggle_api()
        api.dataset_download_files(dataset_id, path=save_dir, unzip=True)
        print("[*] Sukses mendownload dan unzip model dari Kaggle.")
    except Exception as e:
        print(f"[!] Terjadi error saat download Kaggle: {e}")
