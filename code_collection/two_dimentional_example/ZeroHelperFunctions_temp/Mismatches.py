import torch

from ZeroHelperFunctions.Accuracy_fn import accuracy_fn
import matplotlib.pyplot as plt

class Mismatches:
    def __init__(self, test_dataloader_for_repository, mnist_rover):
        self._get_mismatches(test_dataloader_for_repository, mnist_rover)

    def _get_mismatches(self, test_dataloader_for_repository, mnist_rover):
        self.mismatch_data = []
        self.mismatch_targets = []
        self.mismatch_predictions = []
        self.mismatch_probabilities = []
        self.test_acc = 0
        for X, y in test_dataloader_for_repository:
            predictions = mnist_rover.predictions(X)
            # print(predictions.shape)
            # print(y.shape)
            # print(predictions.dtype)
            # print(y.dtype)
            mismatch_indices = (predictions != y).nonzero(as_tuple=True)[0]
            self.mismatch_data.append(X[mismatch_indices])
            self.mismatch_targets.append(y[mismatch_indices])
            self.mismatch_predictions.append(predictions[mismatch_indices])
            self.mismatch_probabilities.append(mnist_rover.temp_probabilites[mismatch_indices])
            
            # print(predictions[:100], y[:100])
            self.test_acc += accuracy_fn(y, predictions)
        self.test_acc = self.test_acc/len(test_dataloader_for_repository)

        if self.mismatch_data:
            self.mismatch_data = torch.cat(self.mismatch_data)
            self.mismatch_targets = torch.cat(self.mismatch_targets)
            self.mismatch_predictions = torch.cat(self.mismatch_predictions)
            self.mismatch_probabilities = torch.cat(self.mismatch_probabilities)
        else:
            self.mismatch_data = torch.tensor([])
            self.mismatch_targets = torch.tensor([])
            self.mismatch_predictions = torch.tensor([])
            self.mismatch_probabilities = torch.tensor([])

        # print("All mismatched data:", self.mismatch_data.shape[:10])
        # print("All mismatched targets:", self.mismatch_targets[:10])
        # print("all mismatched predictions: ", self.mismatch_predictions[:10])
        # print("all mismatched probabilites: ", self.mismatch_probabilities[:10])
    def plot_mismatches(self, start_index, end_index):
        for idx in range(start_index, end_index):
            # Plotting the first mismatched image
            plt.figure(figsize=(10, 5))

            # Image plot
            plt.subplot(1, 2, 1)
            plt.imshow(self.mismatch_data[idx], cmap='gray', vmin=0, vmax=1)
            plt.title('Mismatched Image')

            # Bar chart plot
            plt.subplot(1, 2, 2)
            probabilities = self.mismatch_probabilities[idx].numpy()
            bars = plt.bar(range(10), probabilities, color='blue')

            # Highlight the mismatched target and prediction
            bars[self.mismatch_targets[idx]].set_color('green')
            bars[self.mismatch_predictions[idx]].set_color('red')
            plt.title('Probabilities')
            plt.xlabel('Label')
            plt.ylabel('Probability')

            plt.show()
    