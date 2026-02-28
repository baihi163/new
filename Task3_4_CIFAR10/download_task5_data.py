import os
import requests
import zipfile
from tqdm import tqdm

# 数据集 URL
URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
ZIP_FILE = "PennFudanPed.zip"

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    print("Download complete!")

def extract_file(filename):
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete!")

if __name__ == "__main__":
    # 1. 下载
    download_file(URL, ZIP_FILE)
    
    # 2. 解压
    if os.path.exists("PennFudanPed"):
        print("Dataset folder 'PennFudanPed' already exists.")
    else:
        extract_file(ZIP_FILE)
    
    print("\n✅ Data preparation finished! You should see a 'PennFudanPed' folder.")