
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch




torch.manual_seed(42)
from SyntheticHelperFunctions_paper.GetStandardData import preprocessed_synthetic_data
train_data, test_data, x_test_for_plotting, y_test_for_plotting = preprocessed_synthetic_data(n_train_samples=6000,
                                                    n_test_samples=900,
                                                    n_features=2,
                                                    n_classes=1,
                                                    random_state=42)



from ZeroHelperFunctions_paper.DataLoadersForZero import DataLoadersForZero
dl = DataLoadersForZero(train_data=train_data,
                    test_data=test_data,
                    data_shape_of_data_point=(2))


BATCH_SIZE = 32
dl.make_dataloaders(batch_size=BATCH_SIZE, 
                    n_train_zeros=6_000, 
                    n_test_zeros=900,
                    label_for_zero=1)

zero_class_data, zero_class_labels = dl.zero_data_for_printing, dl.zero_labels_for_printing
zero_class_data.shape


dl.generate_zero_class_dataloader(n_zeros=6_000,
                                batch_size=BATCH_SIZE,
                                label=1)



torch.manual_seed(42)
from ZeroHelperFunctions_paper.zeroTrainer import ZeroTrainer


from Networks.networks import FullyConnectedNet



NUM_DIMENSIONS = 2
NUM_EPOCHS = 100
learning_rate = 0.005

zero_model = FullyConnectedNet(input_dim=NUM_DIMENSIONS,
                    layer1_dim=NUM_DIMENSIONS * 10,
                    layer2_dim=NUM_DIMENSIONS * 5,
                    layer3_dim=11*10,
                    num_classes=2)



# Import PyTorch
import torch
from torch import nn

torch.manual_seed(42)
zero_trainer = ZeroTrainer(model=zero_model,
                        number_of_non_zero_classes=1,
                        train_dl=dl.train0_dataloader,
                        test_dl=dl.test_dataloader,
                        purity_fact_dl=dl.test0_dataloader,
                        zero_dl=dl.zero_dataloader,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.SGD(params=zero_model.parameters(), lr=learning_rate),
                        label_of_zero_class=1,
                        device="cpu" if torch.cuda.is_available() else "cpu",
                        plot_data = dl.combined_test_data,
                        plot_targets = dl.combined_test_targets,
                        zero_exists=1)



zero_trainer.train(epochs=NUM_EPOCHS)



from ZeroHelperFunctions_paper import plots




from ZeroHelperFunctions_paper.helper_functions import plot_decision_boundary, plot_decision_boundary_non_zero



import matplotlib.pyplot as plt
x_range = (-6.5, 1.2)
y_range = (5, 13.5)
fig = plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
markers = ['P', '^', 's', 'D', 'X', '*']  # Add more shapes if needed
for cls in range(2):
    print("wtf", cls)
    if cls == 2 - 1:  # Check if it's the last class
        print("we are in")
        plt.scatter(
            dl.combined_test_data[dl.combined_test_targets == cls, 0],  # X-coordinate for class `cls`
            dl.combined_test_data[dl.combined_test_targets == cls, 1],  # Y-coordinate for class `cls`
            color='#008080',  # Fill color for the last class
            label=f'Zero-Vector Class', 
            marker=markers[cls % len(markers)], 
            edgecolor='#008080',  # Specific edge color for the last class
            alpha=0.9,  # Optional: Different transparency for the last class
            linewidths=1,  # Optional: Thicker edge for better visibility
            s=5
        )
    else:  # For all other classes
        plt.scatter(
            dl.combined_test_data[dl.combined_test_targets == cls, 0],  # X-coordinate for class `cls`
            dl.combined_test_data[dl.combined_test_targets == cls, 1],  # Y-coordinate for class `cls`
            color='none',  # Uniform transparent color
            label=f'Class {cls}', 
            marker=markers[cls % len(markers)], 
            edgecolor='black',  # Black edge for other classes
            alpha=0.7,  # Transparency for other classes
            linewidths=1,
        )
plt.xticks([])
plt.yticks([])
plt.xlim(x_range)
plt.ylim(y_range)
legend = plt.legend(fontsize=12, loc='lower right')
legend.get_frame().set_alpha(1.0)
plt.subplot(1, 2, 2)
# plt.title("Decision Boundaries")
plot_decision_boundary(zero_model, dl.combined_test_data, dl.combined_test_targets, n_classes=2)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0.005, bottom=0.014, right=0.995, top=0.986, wspace=0.052, hspace=0.21)
plt.xlim(x_range)
plt.ylim(y_range)
plt.show()




