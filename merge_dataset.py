
import os
import shutil
import hashlib

source = "dataset/PlantVillage"
destination = "dataset"

def get_hash(path):
    sha = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

merged = 0
duplicates = 0

for class_name in os.listdir(source):
    source_folder = os.path.join(source, class_name)
    destination_folder = os.path.join(destination, class_name)

    if not os.path.isdir(source_folder):
        continue

    os.makedirs(destination_folder, exist_ok=True)

    existing_hashes = {
        get_hash(os.path.join(destination_folder, name))
        for name in os.listdir(destination_folder)
        if os.path.isfile(os.path.join(destination_folder, name))
    }

    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)

        if not os.path.isfile(source_file):
            continue

        file_hash = get_hash(source_file)

        if file_hash in existing_hashes:
            duplicates += 1
            continue

        shutil.copy2(
            source_file,
            os.path.join(destination_folder, filename)
        )
        existing_hashes.add(file_hash)
        merged += 1

print("New images merged:", merged)
print("Duplicate images skipped:", duplicates)
print("Dataset merge completed!")
