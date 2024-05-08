import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

IMG_PATH = "wikiart/Abstract_Expressionism"

# iterate over dir and get list of files
img_paths = pd.Series(os.listdir(IMG_PATH))
img_paths = img_paths[img_paths.str.contains(".jpg")].reset_index(drop=True)
imgs = []

# read 100 images
for i in range(100):
    img = plt.imread(os.path.join(IMG_PATH, img_paths[i]))
    imgs.append(img)


size = 64
# resize and crop images to 64x64
imgs_resized = []
for img in imgs:
    img_resized = jax.image.resize(img, (size, size, 3), method="nearest")
    imgs_resized.append(img_resized)
plt.show()

# plot 4 images in 2x2 grid
fig, axs = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        axs[i, j].imshow(imgs_resized[i*2+j])
        axs[i, j].axis("off")

plt.show()