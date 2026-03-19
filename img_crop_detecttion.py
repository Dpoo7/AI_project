# ==============================
# CROP DETECTOR (FINAL FIXED)
# ==============================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os, cv2, random, shutil

# ==============================
# 1. DOWNLOAD DATASET
# ==============================
!wget -q https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
!tar -xzf flower_photos.tgz

source = "/content/flower_photos"
orig_path = "/content/original"
crop_path = "/content/cropped"

os.makedirs(orig_path, exist_ok=True)
os.makedirs(crop_path, exist_ok=True)

# ==============================
# 2. COLLECT IMAGES
# ==============================
all_images = []

for folder in os.listdir(source):
    folder_path = os.path.join(source, folder)

    if not os.path.isdir(folder_path):
        continue

    for img in os.listdir(folder_path):
        if img.endswith(('.jpg','.jpeg','.png')):
            all_images.append(os.path.join(folder_path, img))

selected = random.sample(all_images, 300)   #  more data

for img in selected:
    shutil.copy(img, orig_path)

print("Original ready ")

# ==============================
# 3. CREATE CROPPED (STRONG)
# ==============================
for img_name in os.listdir(orig_path):
    img = cv2.imread(os.path.join(orig_path, img_name))

    if img is None:
        continue

    h, w, _ = img.shape

    #  strong crop (better learning)
    crop_x = random.randint(w//3, w//2)
    crop_y = random.randint(h//3, h//2)

    cropped = img[crop_y:h-crop_y, crop_x:w-crop_x]

    # safety check
    if cropped is None or cropped.shape[0] < 30 or cropped.shape[1] < 30:
        continue

    # resize back
    cropped = cv2.resize(cropped, (w, h))

    cv2.imwrite(os.path.join(crop_path, img_name), cropped)

print("Cropped ready ✅")

# ==============================
# 4. DATASET READY
# ==============================
dataset = "/content/dataset"

os.makedirs(dataset+"/original", exist_ok=True)
os.makedirs(dataset+"/cropped", exist_ok=True)

for i in os.listdir(orig_path):
    shutil.copy(orig_path+"/"+i, dataset+"/original/"+i)

for i in os.listdir(crop_path):
    shutil.copy(crop_path+"/"+i, dataset+"/cropped/"+i)

print("Dataset ready ✅")

# ==============================
# 5. LOAD DATA
# ==============================
img_size = 64

data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    image_size=(img_size, img_size),
    batch_size=32
)

data = data.map(lambda x,y: (x/255.0, y))

# ==============================
# 6. MODEL (IMPROVED)
# ==============================
model = models.Sequential([
    layers.Conv2D(32,3,activation='relu',input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation='relu'),   # extra layer
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==============================
# 7. TRAIN (LONGER)
# ==============================
model.fit(data, epochs=8)   #  better training

# ==============================
# 8. TEST WITH YOUR IMAGE
# ==============================
from google.colab import files
from tensorflow.keras.preprocessing import image
while True:
    print("\nUpload Image (Press cancel to stop) ")

    uploaded = files.upload()

    if len(uploaded) == 0:
        print("Stopped ")
        break

    for file in uploaded.keys():
        img = image.load_img(file, target_size=(64,64))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]

        print("\nRESULT ")

        if pred > 0.5:
             print(f"Original Image  ({round(pred*100,2)}%)")
        else:
            print(f"Cropped Image  ({round((1-pred)*100,2)}%)")
