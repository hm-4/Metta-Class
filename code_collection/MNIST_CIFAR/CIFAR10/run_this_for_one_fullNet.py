#!/usr/bin/env python
# coding: utf-8

import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import torch




torch.manual_seed(42)
from Cifar10HelperFunctions.GetStandardData import preprocessed_cifar10
train_data, test_data = preprocessed_cifar10()



torch.manual_seed(42)

from ZeroHelperFunctions.DataLoadersForZero import DataLoadersForZero
dl = DataLoadersForZero(train_data=train_data,
                        test_data=test_data,
                        image_shape=(3, 32, 32)
                        )




BATCH_SIZE = 16



torch.manual_seed(42)

dl.make_dataloaders(batch_size=BATCH_SIZE, 
                    n_train_zeros=50_000, 
                    n_test_zeros=10_000,
                    label_for_zero=10)



torch.manual_seed(42)

dl.generate_zero_class_dataloader(n_zeros=50_000,
                                  batch_size=BATCH_SIZE,
                                  label=10)



dl.check_dataloader(dl.zero_dataloader)
dl.check_dataloader(dl.train0_dataloader)
dl.check_dataloader(dl.train_dataloader)
dl.check_dataloader(dl.test_dataloader)
dl.check_dataloader(dl.test0_dataloader)



dl.check_elements_of_train_data_if_tensors()



dl.check_elements_of_zero_class_if_tensors()


dl.describe_train_data()



dl.describe_zero_class_data()



dl.show_border_images_of_combined_data(50_000)



torch.manual_seed(42)
from ZeroHelperFunctions.zeroTrainer import ZeroTrainer
from Networks.networks import AllClassVGGNet




torch.manual_seed(42)
NUM_EPOCHS = 50
learning_rate = 0.001

zero_model = AllClassVGGNet(out_nodes=11)

simple_model = AllClassVGGNet(out_nodes=10)



# Import PyTorch
import torch
from torch import nn

torch.manual_seed(42)
zero_trainer = ZeroTrainer(model=zero_model,
                        number_of_non_zero_classes=10,
                        train_dl=dl.train0_dataloader,
                        test_dl=dl.test_dataloader,
                        purity_fact_dl=dl.test0_dataloader,
                        zero_dl=dl.zero_dataloader,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.SGD(params=zero_model.parameters(), lr=learning_rate),
                        label_of_zero_class=10,
                        device="cuda" if torch.cuda.is_available() else "cpu")

simple_trainer = ZeroTrainer(model=simple_model,
                        number_of_non_zero_classes=10,
                        train_dl=dl.train_dataloader,
                        test_dl=dl.test_dataloader,
                        purity_fact_dl=dl.test0_dataloader,
                        zero_dl=dl.zero_dataloader,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.SGD(params=simple_model.parameters(), lr=learning_rate),
                        label_of_zero_class=10,
                        device="cuda" if torch.cuda.is_available() else "cpu")



zero_trainer.train(epochs=NUM_EPOCHS)


from ZeroHelperFunctions import plots



plots.plot_pf(zero_trainer.purities, title="Zero-Inclusive-Net", path="plots/cifar_full_zero/purity")
plots.plot_of(zero_trainer.occupancy, title="Zero-Inclusive-Net", path="plots/cifar_full_zero/occupancy")
plots.plot_train_test_losses(zero_trainer.train_loss,
                             zero_trainer.test_loss, title="Zero-Inclusive-Net", path="plots/cifar_full_zero/loss")
plots.plot_train_test_accs(zero_trainer.train_acc,
                           zero_trainer.test_acc, title="Zero-Inclusive-Net", path="plots/cifar_full_zero/acc")



simple_trainer.train(epochs=NUM_EPOCHS)




plots.plot_pf(simple_trainer.purities, title="Zero-Exclusive-Net", path="plots/cifar_full_no_zero/purity")
plots.plot_of(simple_trainer.occupancy, title="Zero-Exclusive-Net", path="plots/cifar_full_no_zero/occupancy")
plots.plot_train_test_losses(simple_trainer.train_loss,
                             simple_trainer.test_loss, title="Zero-Exclusive-Net", path="plots/cifar_full_no_zero/loss")
plots.plot_train_test_accs(simple_trainer.train_acc,
                           simple_trainer.test_acc, title="Zero-Exclusive-Net", path="plots/cifar_full_no_zero/acc")


# plots.plot_pf(simple_trainer.purities, title="Zero-Exclusive-Net")
# plots.plot_of(simple_trainer.occupancy, title="Zero-Exclusive-Net")
# plots.plot_train_test_losses(simple_trainer.train_loss, simple_trainer.test_loss, title="Zero-Exclusive-Net")
# plots.plot_train_test_accs(simple_trainer.train_acc, simple_trainer.test_acc, title="Zero-Exclusive-Net")

