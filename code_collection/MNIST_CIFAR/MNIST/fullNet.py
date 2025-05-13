#!/usr/bin/env python
# coding: utf-8

# In[1]:

print("reading here. ", flush=True)
import sys
import os
import gc
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('/home/harikrishnam/the_project/zero0.3')

# In[2]:

print("gettning that data!! ", flush=True)
from MnistHelperFunctions.get_standard_data import preprocessed_mnist
train_data, test_data = preprocessed_mnist()


# In[3]:


from ZeroHelperFunctions.DataLoadersForZero import DataLoadersForZero
dl = DataLoadersForZero(train_data=train_data,
                        test_data=test_data,
                        image_shape=(1, 28, 28))


# In[4]:


dl.make_dataloaders(batch_size=256, 
                    n_train_zeros=60_000, 
                    n_test_zeros=10_000,
                    label_for_zero=10)


# In[5]:


dl.show_border_images_of_combined_data(60_000)

print("1 ", flush=True)
# In[6]:


dl.generate_zero_class_dataloader(10000, 256)


# In[7]:


dl.check_dataloader(dl.zero_dataloader)
dl.check_dataloader(dl.train0_dataloader)
dl.check_dataloader(dl.train_dataloader)
dl.check_dataloader(dl.test_dataloader)
dl.check_dataloader(dl.test0_dataloader)

print("2 ", flush=True)
# In[8]:


# dl.check_elements_of_train_data_if_tensors()


# In[9]:


# dl.check_elements_of_zero_class_if_tensors()


# In[10]:


dl.describe_train_data()


# In[11]:


dl.describe_zero_class_data()


# In[12]:


from ZeroHelperFunctions.zeroTrainer import ZeroTrainer
from Networks.networks import FullyConnectedNet


# In[13]:


NUM_DIMENSIONS = 28*28
NUM_EPOCHS = 100
learning_rate = 0.01

zero_model = FullyConnectedNet(input_dim=NUM_DIMENSIONS,
                    layer1_dim=NUM_DIMENSIONS * 10,
                    layer2_dim=NUM_DIMENSIONS * 5,
                    layer3_dim=11*10,
                    num_classes=11)

simple_model = FullyConnectedNet(input_dim=NUM_DIMENSIONS,
                    layer1_dim=NUM_DIMENSIONS * 10,
                    layer2_dim=NUM_DIMENSIONS * 5,
                    layer3_dim=11*10,
                    num_classes=10)


# In[14]:

print("model created ", flush=True)
# Import PyTorch
import torch
from torch import nn

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

print("trainer created ", flush=True)
# In[15]:


zero_trainer.device


# In[16]:

print("training ", flush=True)
zero_trainer.train(epochs=NUM_EPOCHS)

print("training done on mnist", flush=True)
# In[17]:


from ZeroHelperFunctions import plots


# In[18]:


plots.plot_pf(zero_trainer.purities, title="Zero-Inclusive-Net")
plots.plot_of(zero_trainer.occupancy, title="Zero-Inclusive-Net")
plots.plot_train_test_losses(zero_trainer.train_loss,
                             zero_trainer.test_loss, title="Zero-Inclusive-Net")
plots.plot_train_test_accs(zero_trainer.train_acc,
                           zero_trainer.test_acc, title="Zero-Inclusive-Net")


# In[19]:


simple_trainer.train(epochs=NUM_EPOCHS)


# In[ ]:


plots.plot_pf(simple_trainer.purities, title="Zero-Exclusive-Net")
plots.plot_of(simple_trainer.occupancy, title="Zero-Exclusive-Net")
plots.plot_train_test_losses(simple_trainer.train_loss, simple_trainer.test_loss, title="Zero-Exclusive-Net")
plots.plot_train_test_accs(simple_trainer.train_acc, simple_trainer.test_acc, title="Zero-Exclusive-Net")

