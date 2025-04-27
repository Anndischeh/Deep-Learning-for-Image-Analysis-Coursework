import os
import subprocess  # For running external commands
import zipfile
import requests
from tqdm import tqdm

# Define the base directory
base_dir = "/content/"  # Use a Windows path
# Create the directories
os.makedirs(os.path.join(base_dir, "input"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "working", "lossandmap"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "working", "plots"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "working", "weights"), exist_ok=True)


def download_and_extract(url, dest_dir):
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)

    print(f"Downloading {url} to {filepath}")

    # Download the file using requests with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

    print(f"Finished downloading {url}")

    # Extract the zip file
    print(f"Extracting {filepath} to {dest_dir}")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            print(f"Finished extracting {filepath}")
    except zipfile.BadZipFile:
        print(f"Error extracting {filepath}")
        return False

    # Remove the zip file
    try:
        os.remove(filepath)
        print(f"Removed {filepath}")
    except OSError as e:
        print(f"Error removing {filepath}: {e}")
        return False

    return True

# Download and extract annotations
print("Downloading annotations...")
download_and_extract("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", os.path.join(base_dir, "input"))
print("Annotations download complete.")

# Download and extract image_info
print("Downloading image info...")
download_and_extract("http://images.cocodataset.org/annotations/image_info_test2017.zip", os.path.join(base_dir, "input"))
print("Image info download complete.")

# Download and extract train data
print("Downloading train data...")
download_and_extract("http://images.cocodataset.org/zips/train2017.zip", os.path.join(base_dir, "input"))
print("Train data download complete.")

# Download and extract validation data
print("Downloading validation data...")
download_and_extract("http://images.cocodataset.org/zips/val2017.zip", os.path.join(base_dir, "input"))
print("Validation data download complete.")

# Download and extract test data
print("Downloading test data...")
download_and_extract("http://images.cocodataset.org/zips/test2017.zip", os.path.join(base_dir, "input"))
print("Test data download complete.")


# List the contents of the input directory
print("Contents of input directory:")
for item in os.listdir(os.path.join(base_dir, "input")):
    print(item)