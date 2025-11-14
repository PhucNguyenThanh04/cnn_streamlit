import os
import shutil
import random
from sklearn.model_selection import train_test_split

source_dir = 'dataset_kidney'
target_dir = 'Ultrasound_Stone_No_Stone'

if os.path.exists(target_dir):
    os.makedirs(target_dir)

# Tỉ lệ chia dữ liệu
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Tạo thư mục đích
for split in ['train', 'val', 'test']:
    for category in ['Normal', 'stone']:
        os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

# Lặp qua từng lớp (Normal, stone)
for category in ['Normal', 'stone']:
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)

    # Shuffle ngẫu nhiên danh sách ảnh
    random.shuffle(images)

    # Chia tập train, val, test
    train_files, temp_files = train_test_split(images, test_size=(1 - train_ratio))
    val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (test_ratio + val_ratio)))

    # Copy file vào thư mục tương ứng
    for file in train_files:
        shutil.copy(os.path.join(category_path, file),
                    os.path.join(target_dir, 'train', category, file))

    for file in val_files:
        shutil.copy(os.path.join(category_path, file),
                    os.path.join(target_dir, 'val', category, file))

    for file in test_files:
        shutil.copy(os.path.join(category_path, file),
                    os.path.join(target_dir, 'test', category, file))

print("✅ Dataset đã được chia thành công!")
