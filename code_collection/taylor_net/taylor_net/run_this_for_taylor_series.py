#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_blobs
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ZeroNet import ZeroNet, SimpleNet, generate_zero_data


import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


# In[2]:


NUM_DATASAMPLES = 1000
NUM_CLASSES = 1
NUM_DIMENSIONS = 2
X, y = make_blobs(n_samples=NUM_DATASAMPLES,
                  centers=NUM_CLASSES,
                  n_features=NUM_DIMENSIONS,
                  random_state=42)
print(y[:5])
y = y+1.
print(y[:5])

print(type(X), type(y), X.dtype, y.dtype)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(type(X), type(y), X.shape, y.shape, X.dtype, y.dtype)


# In[3]:


plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
y[:5]


# In[4]:


min_pt_of_each_dim_of_space = torch.min(X, dim=0)
max_pt_of_each_dim_of_space = torch.max(X, dim=0)
min_pt_of_each_dim_of_space[0], max_pt_of_each_dim_of_space[0]

zero_generator = generate_zero_data(min_pt_of_each_dim_of_space[0],
                                    max_pt_of_each_dim_of_space[0])

X0 = zero_generator.sample((NUM_DATASAMPLES,))
y0 = torch.zeros((NUM_DATASAMPLES,), dtype=torch.float)
X0[:5], type(X0), X0.shape, X[:5], type(X), X.shape, y[:5], y0[:5], y.shape, y0.shape


# In[5]:


X1_and_0 = torch.cat((X, X0), dim=0)
y1_and_0 = torch.cat((y, y0))
X1_and_0.shape, y1_and_0.shape


# In[6]:


y1_and_0[:100]


# In[7]:


plt.scatter(x=X1_and_0[:, 0],
            y=X1_and_0[:, 1],
            c=y1_and_0,
            cmap=plt.cm.RdYlBu)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X1_and_0, y1_and_0,
                                                    test_size=0.2,
                                                    random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:





# In[9]:


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataset[0], X_train[0], y_train[0]


# In[10]:


# Step 5: Create DataLoaders for batch training

BATCH_SIZE = 8
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)}, batches of size {BATCH_SIZE}")
print(f"Length of train dataloader: {len(test_dataloader)}, batches of size {BATCH_SIZE}")


# In[11]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[12]:


from TaylorNet import TaylorNet


# In[13]:


torch.manual_seed(42)
polynomial_basis_model = TaylorNet(input_dim=2,
                                   num_taylor_monomials=3)
polynomial_basis_model.to(device)


# In[14]:


list(polynomial_basis_model.parameters())


# In[15]:


from helper_functions import accuracy_fn

loss_fn = nn.BCEWithLogitsLoss()

learning_rates = {
    'monomial_0': 0.001,
    'monomial_1': 0.0001,
    'monomial_2': 0.00001
}

# Prepare optimizer parameter groups
param_groups = []
for name, param in polynomial_basis_model.taylor_coefficients.items():
    param_groups.append({'params': [param], 'lr': learning_rates[name]})

# Add the Taylor offsets with a different learning rate
param_groups.append({'params': [polynomial_basis_model.taylor_offsets], 'lr': 0.01})


optimizer = torch.optim.SGD(param_groups)

# Verify the parameter groups
print(optimizer)


# In[16]:


from taylor_training_uniclass import train_loop


# In[22]:


epochs = 3000
train_loop(epochs, 
           train_dataloader, 
           test_dataloader,
           model=polynomial_basis_model,
           loss_fn=loss_fn,
           optimizer=optimizer,
           accuracy_fn=accuracy_fn,
           device=device)


# In[ ]:


from helper_functions import plot_predictions, plot_decision_boundary
# Enable LaTeX rendering in Matplotlib
import matplotlib as mpl
# Enable LaTeX rendering
mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}"  # Include amsmath
})

def display_text_results(results_text):
    """Display text results on the current subplot."""
    plt.axis("off")  # Turn off axes
    plt.text(0.1, 0.5, results_text, fontsize=18, verticalalignment='center',
             bbox=dict(facecolor='white', edgecolor='black', alpha=1))

# Prepare results text
# Split the third term into multiple lines for wrapping
result_list = list(polynomial_basis_model.named_parameters())
x_0 = result_list[0][1].data.numpy()
b = result_list[1][1].data.numpy()
l1 = result_list[2][1].data.numpy()
l2 = result_list[3][1].data.numpy()
print(x_0, b)

results_text = (
    r"Coefficients:" "\n"
    "\n"
    r"$\mathbf{x}=\hat{\mathbf{x}}-\mathbf{x}_0$""\n"
    r"$h(\mathbf{x}) = m_0 + \mathbf{x}^T \mathbf{m}_1 + \mathbf{x}^T \mathbf{m}_2 \mathbf{x}$"
    "\n\n"
    r"$\mathbf{x}_0$:" f" {x_0}\n"
    r"$m_0$:" f" {b}\n"
    r"$\mathbf{m}_1$:" f" {l1}\n"
    r"$\mathbf{m}_2$:" f" {l2}"
)

# Create figure
plt.figure(figsize=(12, 6))

# Subplot 1: Sine wave
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(polynomial_basis_model, X_train, y_train, n_classes=2)


# Subplot 2: Text results
plt.subplot(1, 2, 2)
# plt.title("Coefficients")
display_text_results(results_text)

# Show plots
# plt.tight_layout()
plt.subplots_adjust(left=0.049, bottom=0.073, right=0.945, top=0.928, wspace=0.05, hspace=0.2)
plt.show()



# In[109]:


polynomial_basis_model.taylor_offsets


# In[110]:


params = list(polynomial_basis_model.named_parameters())


# In[111]:


for name, param in params:
    print(f'{name}: {param.data.numpy()}')


# In[112]:


list(polynomial_basis_model.named_parameters())[0][1].data


# In[ ]:




