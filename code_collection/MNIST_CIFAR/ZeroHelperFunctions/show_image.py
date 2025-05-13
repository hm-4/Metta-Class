import matplotlib.pyplot as plt
import numpy as np

# Define a function to show an image

import numpy as np
import matplotlib.pyplot as plt

def show_one_color_image(img):
    # Check if the input is a tensor, and convert it to numpy if necessary
    if isinstance(img, np.ndarray):
        npimg = img
    else:
        npimg = img.detach().cpu().numpy()  # Convert tensor to numpy if it's a tensor

    # Ensure the image has the correct shape for imshow (H, W, C)
    npimg = np.transpose(npimg, (1, 2, 0))  # For (C, H, W) -> (H, W, C)

    plt.imshow(npimg)
    plt.show()

    
    

def show_one_grayscale_image(image, title='Sample'):
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.show()
    

def show_color_images_in_grid(images, rows, cols):
    """
    Display a grid of images.

    Parameters:
    images (numpy array or tensor): A 4D array or tensor containing
    images, with shape (num_images, channels, width, height).
    images must be in the valid range [0, 1]
    rows (int): Number of rows in the grid.
    cols (int): Number of columns in the grid.
    """
    num_images = images.shape[0]
    channels = images.shape[1]
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i]
            if channels == 1:
                img = img[0, :, :]
                ax.imshow(img, cmap='gray')
            else:
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_grayscale_images_in_grid(images, rows, cols):
    """
    Display a grid of grayscale images.

    Parameters:
    images (numpy array or tensor): A 4D array or tensor containing images, with shape (num_images, channels, width, height).
    rows (int): Number of rows in the grid.
    cols (int): Number of columns in the grid.
    """
    num_images = images.shape[0]
    channels = images.shape[1]
    
    if channels != 1:
        raise ValueError("Each image must have only one channel for grayscale images.")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i, 0, :, :]  # Extract the grayscale image
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()




def show_channels(image):
    # Convert tensor to numpy array
    npimg = image.numpy()
    # Separate the channels
    R = 1 - npimg[0, :, :]
    G = 1 - npimg[1, :, :]
    B = 1 - npimg[2, :, :]
    
    # Plot the original image and each channel
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    
    axs[0].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[0].set_title('Original Image')
    
    axs[1].imshow(R, cmap='Reds')
    axs[1].set_title('Red Channel')
    
    axs[2].imshow(G, cmap='Greens')
    axs[2].set_title('Green Channel')
    
    axs[3].imshow(B, cmap='Blues')
    axs[3].set_title('Blue Channel')
    
    for ax in axs:
        ax.axis('off')
    
    plt.show()