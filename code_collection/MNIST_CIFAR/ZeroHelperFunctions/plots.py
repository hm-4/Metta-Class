import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


import os
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

def plot_pf(tensor, title="Zero-Exclusive-Net", path="plots/purity"):
    """
    Plots each column of a tensor on the same figure with legends and saves the plot as a PDF.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (n, c).
        title (str): The title of the plot and the folder name.
        path (str): The base directory where the plot will be saved.
    """
    # Sanitize the title for valid folder names
    sanitized_title = title.replace(" ", "_").replace("/", "-")  # Replace spaces and slashes for valid file names
    path_with_title = os.path.join(path, sanitized_title)
    
    # Ensure the directory exists
    if not os.path.exists(path_with_title):
        os.makedirs(path_with_title)
    
    # Ensure the tensor is on CPU and convert to NumPy
    tensor = tensor.cpu().numpy()
    
    # Get the number of columns
    n, c = tensor.shape
    
    # Create a new figure
    plt.figure(figsize=(8, 6))
    
    # Plot each column on the same axes
    for i in range(c):
        plt.plot(tensor[:, i], marker='o', markersize=2, linewidth=0.5, label=f'class-{i}')
    
    # Add title, labels, legend, and grid
    plt.title(f'Purity Factors ({title})', fontsize=16)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure as a PDF in the title folder
    file_path = os.path.join(path_with_title, f"{sanitized_title}_purity_factors.pdf")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Plot saved to {file_path}")



import os
import matplotlib.pyplot as plt

def plot_of(occupancy, title="Zero-Exclusive-Net", path="plots/occupancy"):
    """
    Plots occupancy factor and saves the plot in a dynamically created title-based folder.
    
    Args:
        occupancy (list or array): The occupancy factor values.
        title (str): The title of the plot and the folder name.
        path (str): The base directory where the plot will be saved.
    """
    # Sanitize the title for valid folder names
    sanitized_title = title.replace(" ", "_").replace("/", "-")  # Replace spaces and slashes for valid file names
    path_with_title = os.path.join(path, sanitized_title)
    
    # Ensure the directory exists
    if not os.path.exists(path_with_title):
        os.makedirs(path_with_title)
    
    plt.figure(figsize=(8, 6))
    plt.plot(occupancy, marker='o', markersize=2, linewidth=0.5)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epochs')
    plt.title(f'Occupancy Factor ({title})', fontsize=16)
    
    # Save the figure as a PDF in the title folder
    file_path = os.path.join(path_with_title, f"{sanitized_title}_occupancy_factor.pdf")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Plot saved to {file_path}")


def plot_train_test_losses(train_losses, test_losses, title="Zero-Exclusive-Net", path="plots/losses"):
    """
    Plots training and testing losses and saves the plot in a dynamically created title-based folder.
    
    Args:
        train_losses (list or array): Training loss values.
        test_losses (list or array): Testing loss values.
        title (str): The title of the plot and the folder name.
        path (str): The base directory where the plot will be saved.
    """
    # Sanitize the title for valid folder names
    sanitized_title = title.replace(" ", "_").replace("/", "-")  # Replace spaces and slashes for valid file names
    path_with_title = os.path.join(path, sanitized_title)
    
    # Ensure the directory exists
    if not os.path.exists(path_with_title):
        os.makedirs(path_with_title)
    
    plt.figure(figsize=(8, 6))
    train_losses = [t.item() for t in train_losses]
    test_losses = [t.item() for t in test_losses]
    plt.plot(train_losses, marker='o', markersize=2, linewidth=0.5, label='Train Losses')
    plt.plot(test_losses, marker='x', markersize=2, linewidth=0.5, label='Test Losses')
    plt.legend()
    plt.xlabel('epoch')
    plt.title(f'Losses ({title})', fontsize=16)
    
    # Save the figure as a PDF in the title folder
    file_path = os.path.join(path_with_title, f"{sanitized_title}_losses.pdf")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Plot saved to {file_path}")




def plot_train_test_accs(train_accs, test_accs, title="Zero-Exclusive-Net", path="plots/accuracies"):
    """
    Plots training and testing accuracies and saves the plot with a title-based directory and file name.
    
    Args:
        train_accs (list or array): Training accuracy values.
        test_accs (list or array): Testing accuracy values.
        title (str): The title of the plot and the folder name.
        path (str): The base directory where the plot will be saved.
    """
    # Sanitize the title for valid folder names
    sanitized_title = title.replace(" ", "_").replace("/", "-")  # Replace spaces and slashes for valid file names
    path_with_title = os.path.join(path, sanitized_title)
    
    # Ensure the directory exists
    if not os.path.exists(path_with_title):
        os.makedirs(path_with_title)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, marker='o', markersize=2, linewidth=0.5, label='Train accuracy')
    plt.plot(test_accs, marker='x', markersize=2, linewidth=0.5, label='Test accuracy')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.title(f'Accuracies ({title})', fontsize=16)
    plt.tight_layout()

    # Save the figure as a PDF in the title folder
    file_path = os.path.join(path_with_title, "accuracies.pdf")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Plot saved to {file_path}")
