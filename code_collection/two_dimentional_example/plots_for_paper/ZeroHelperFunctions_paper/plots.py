import matplotlib.pyplot as plt

def plot_pf(tensor):
    """
    Plots each column of a tensor on the same figure with legends.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (n, c).
    """
    # Ensure the tensor is on CPU and convert to NumPy
    tensor = tensor.cpu().numpy()
    
    # Get the number of columns
    n, c = tensor.shape
    
    # Create a new figure
    plt.figure(figsize=(10, 8))
    
    # Plot each column on the same axes
    for i in range(c):
        plt.plot(tensor[:, i], marker='o', markersize=2, linewidth=0.5, label=f'class-{i}')
    
    # Add title, labels, legend, and grid
    plt.title('Purity factors')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epochs')
    # plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_of(occupancy):
    plt.figure(figsize=(10, 8))
    plt.plot(occupancy, marker='o', markersize=2, linewidth=0.5)
    plt.ylim(-0.1, 1.1)
    # plt.xlim(0, 20)
    plt.xlabel('Epochs')
    # plt.ylabel('occupancy')
    plt.title('Occupancy')
    plt.show()

def plot_train_test_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 8))
    train_losses = [t.item() for t in train_losses]
    test_losses = [t.item() for t in test_losses]
    plt.plot(train_losses, marker='o', markersize=2, linewidth=0.5, label='Train Losses w/ zero class')
    plt.plot(test_losses, marker='x', markersize=2, linewidth=0.5, label='Test Losses w/o zero class')
    plt.legend()
    plt.xlabel('epoch')
    # plt.ylabel('occupancy')
    plt.title('Losses')
    plt.show()

def plot_train_test_accs(train_accs, test_accs):
    plt.figure(figsize=(10, 8))
    plt.plot(train_accs, marker='o', markersize=2, linewidth=0.5, label='Train accuracy w/ zero class')
    plt.plot(test_accs, marker='x', markersize=2, linewidth=0.5, label='Test accuracy w/o zero class')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylim(-5, 105)
    # plt.ylabel('occupancy')
    plt.title('Accuracy')
    plt.show()