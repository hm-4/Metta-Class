import torch
from torch import nn
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = "cpu"):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        

        # 2. Calculate loss
        # print(y_pred.shape, y.shape)
        # print(y_pred, y)
        # print(torch.round(torch.sigmoid(y_pred)))
        loss = loss_fn(y_pred, y.float())
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=torch.round(torch.sigmoid(y_pred))) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = "cpu"):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=torch.round(torch.sigmoid(test_pred))# Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time



# from tqdm.auto import tqdm
from timeit import default_timer as timer
from helper_functions import plot_predictions, plot_decision_boundary
def train_loop(epochs, 
               train_dataloader, test_dataloader,
               model,
               loss_fn,
               optimizer,
               accuracy_fn,
               device):

    torch.manual_seed(42)

    # Measure time
    train_time_start_on_gpu = timer()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn
        )
        test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.title("Train")
        # plot_decision_boundary(zero_model, X_train, y_train)

        # plt.subplot(1, 2, 2)
        # plt.title("Test")
        # plot_decision_boundary(zero_model, X_test, y_test)
        # plt.show()

    train_time_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)
    
    
