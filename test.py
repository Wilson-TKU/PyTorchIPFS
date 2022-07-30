# import ipfshttpclient
# import pytorchipfs
from src.pytorchipfs import __init__
import torch
import torchvision.transforms as transforms

hashes = [
    'bafkreic3aeripksj7a7pnvkiybq3i43hme6pxlmpx7jaokubpz2lfdrvti'
]

client = ipfshttpclient.connect()

# Standard dataset
dataset = pytorchipfs.datasets.IPFSImageTensorDataset(
    client,
    'data', # Where the files will be downloaded
    None, # Don't make assumptions about the image shape
    hashes
)

# Dataset with cropping (to be fed to the model)
cropped_dataset = pytorchipfs.datasets.IPFSImageTensorDataset(
    client,
    'data', # Where the files will be downloaded
    None, # Don't make assumptions about the image shape
    hashes,
    transform=transforms.CenterCrop(32) # Crop the images
)

import matplotlib.pyplot as plt
import numpy as np

for image in dataset:
    # Convert to channel-last Numpy
    image = image.cpu().numpy() / 255
    image = np.transpose(image, (2, 1, 0))

    plt.imshow(image)
    plt.show()