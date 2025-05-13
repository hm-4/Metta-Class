#!/usr/bin/env python
# coding: utf-8


import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch

from torchvision import datasets
from torchvision.transforms import ToTensor


from Cifar10HelperFunctions.GetStandardData import class_wise_preprocessed_cifar10
train_datas, test_datas, test_data_for_repository = class_wise_preprocessed_cifar10()


torch.manual_seed(42)

repository_size = 10
# NUM_DIMENSIONS = 28*28
NUM_EPOCHS = 10
label_for_zero = 0
BATCH_SIZE = 16
learning_rate = 0.001


n_train_zeros = 300_000
n_test_zeros = 100_000

device="cuda" if torch.cuda.is_available() else "cpu"

print("here you are at 1", flush=True)

from ZeroHelperFunctions.DataLoadersForZero import DataLoadersForZero
from Networks.networks import ClassWiseVGGNet
from ZeroHelperFunctions.zeroTrainer import ZeroTrainer

from torch import nn

from ZeroHelperFunctions import plots
from ZeroHelperFunctions.Curiosity import CuriosityRover

mnist_rover = CuriosityRover(device=device)
print("here you are at 2", flush=True)
for cl in range(repository_size):
    print(f"Class-{cl} training ...", flush=True)
    dl = DataLoadersForZero(train_data=train_datas[cl],
                            test_data=test_datas[cl],
                            image_shape=(3, 32, 32))
    dl.make_dataloaders(batch_size=BATCH_SIZE,
                        n_train_zeros=n_train_zeros,
                        n_test_zeros=n_test_zeros,
                        label_for_zero=label_for_zero)
    # dl.show_border_images_of_combined_data(6000)
    dl.generate_zero_class_dataloader(10000, BATCH_SIZE)
    zero_model = ClassWiseVGGNet(out_nodes=2)
    zero_trainer = ZeroTrainer(model=zero_model,
                               number_of_non_zero_classes=1,
                               train_dl=dl.train0_dataloader,
                               test_dl=dl.test_dataloader,
                               purity_fact_dl=dl.test0_dataloader,
                               zero_dl=dl.zero_dataloader,
                               loss_fn=nn.CrossEntropyLoss(),
                               optimizer=torch.optim.SGD(params=zero_model.parameters(), lr=learning_rate),
                               label_of_zero_class=label_for_zero,
                               device=device)
    zero_trainer.train(epochs=NUM_EPOCHS)
    plots.plot_pf(zero_trainer.purities, title=f"class-{cl}", path="plots/cifar_singles/purity")
    plots.plot_of(zero_trainer.occupancy, title=f"class-{cl}", path="plots/cifar_singles/occupancy")
    plots.plot_train_test_losses(zero_trainer.train_loss,
                                 zero_trainer.test_loss, title=f"class-{cl}", path="plots/cifar_singles/losses")
    plots.plot_train_test_accs(zero_trainer.train_acc,
                               zero_trainer.test_acc, title=f"class-{cl}", path="plots/cifar_singles/accs")
    mnist_rover.add_model_for_an_anomaly(zero_model, f"MNIST - {cl}")


from torch.utils.data import DataLoader

test_dataloader_for_repository = DataLoader(test_data_for_repository,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)



from ZeroHelperFunctions.Mismatches import Mismatches
mismatch = Mismatches(test_dataloader_for_repository, mnist_rover)


mismatch.test_acc


from ZeroHelperFunctions.zero_class_images_generator_mnist import ZeroClassDataset



zero_vector_data = ZeroClassDataset(num_samples=10, image_shape=(3, 32, 32), label=label_for_zero)



zero_vector_data.data[0].shape



from ZeroHelperFunctions.show_image import show_one_color_image
show_one_color_image(zero_vector_data.data[0])



xk = zero_vector_data.data[0]

def generate(starting_noise, class_name, step_size, iterations=100, print_freq=10):
    xk = starting_noise
    xk = xk.clone().detach().requires_grad_(True) 
    xk.shape, xk.requires_grad
    zero_model = mnist_rover.learned_anomaly_models[class_name]
    zero_model.eval()
    logits = zero_model(xk.to("cuda" if torch.cuda.is_available() else "cpu"))
    target_logit = logits[:, 1]
    target_logit
    target_logit.backward()
    gradients = xk.grad
    xk = xk + gradients * step_size
    xk[0].shape
    print("wtf")
    show_one_color_image(xk[0].detach().cpu().numpy())
    for i in range(iterations):
        xk = xk.clone().detach().requires_grad_(True) 
        xk.shape, xk.requires_grad
        zero_model.eval()
        logits = zero_model(xk.to("cuda" if torch.cuda.is_available() else "cpu"))
        target_logit = logits[:, 1]
        target_logit
        target_logit.backward()
        gradients = xk.grad
        xk = xk + gradients * step_size
        xk[0].shape
        if i % print_freq == 0:
            print("iteration number: ", i)
            show_one_color_image(xk[0].detach().cpu().numpy())



generate(starting_noise=zero_vector_data.data[0].unsqueeze(0), 
         class_name=0, 
         step_size=0.01, 
         iterations=3001,
         print_freq=500)

