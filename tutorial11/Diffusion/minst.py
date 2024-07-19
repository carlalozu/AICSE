import gzip
import os
import struct
import urllib.request
import array

import jax.numpy as jnp
import matplotlib.pyplot as plt


class MNIST:
    """MNIST dataset class retrieval"""

    def __init__(self):
        self.data = self.assemble_datasets()

    def assemble_datasets(self) -> jnp.ndarray:
        """Downloads and loads MNIST dataset."""
        filename = "train-images-idx3-ubyte.gz"
        url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
        target_dir = os.path.join(os.getcwd(), "data", "mnist")
        url = f"{url_dir}/{filename}"
        target = os.path.join(target_dir, filename)

        if not os.path.exists(target):
            os.makedirs(target_dir, exist_ok=True)
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded {url} to {target}")

        with gzip.open(target, "rb") as fh:
            _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
            shape = (batch, 1, rows, cols)
            return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)

    def normalize(self) -> jnp.ndarray:
        """Normalizes the dataset."""
        print(self.data.shape)
        self.data = (self.data - jnp.mean(self.data)) / jnp.std(self.data)
        return self.data

    def plot(self):
        """Plots the first two images in the dataset."""
        print("Plotting first two images in the dataset.")
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(self.data[0, 0])
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(self.data[1, 0])
        plt.colorbar()
        plt.show()
