#comentari de test 2
from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# download the dataset
url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
datasets.utils.download_and_extract_archive(url, './data')

data_dir = "./data/hymenoptera_data/"

# ResNet input size
input_size = (224,224)

# Just normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


# %% [markdown]
# ### Visualize some images

# %%
import matplotlib.image as mpimg

# show some images
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    idx = np.random.randint(0,len(image_datasets['train'].samples))
    image = mpimg.imread(image_datasets['train'].samples[idx][0])
    plt.imshow(image)
    plt.axis('off');

# ResNet input size
input_size = (224,224)

# Just normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(), # YOUR CODE HERE!
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, and a specified number of epochs
# to train and validate for.
# 

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "test": []}
    losses = {"train": [], "test": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.cpu().detach().numpy())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses

# %% [markdown]
# ## Initialize the ResNet model
# 
# We will use the Resnet18 model (the smaller one), as our dataset is small and only has two classes. When we print the model, we see that the last layer is a fully connected layer as shown below:
# 
# 
#    `(fc): Linear(in_features=512, out_features=1000, bias=True)`
# 
# 

# %%
def initialize_model():
    # Resnet18  
    model = models.resnet18()
    
    input_size = 224

    return model, input_size



# Initialize the model
model, input_size = initialize_model()

# Print the model we just instantiated
print(model)

# %% [markdown]
# ### EXERCISE 2
# 
# Modify the function `initialize_model` so it reinitializes ``model.fc`` to be a Linear layer with 512 input features and 2 output features.
# 

# %%
def initialize_model(num_classes):
    # Resnet18 
    model = models.resnet18()
    
    model.fc = nn.Linear(512,num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model, input_size


# Number of classes in the dataset
num_classes = 2

# Initialize the model
model, input_size = initialize_model(num_classes)

# Print the model we just instantiated
print(model)


# %% [markdown]
# ### Run Training and Validation Step
# 
# Let's start by training the model from scratch. What do you think will happen?
# 

# %%
# Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
model, hist, losses = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# %%
# plot the losses and accuracies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(losses["train"], label="training loss")
ax1.plot(losses["val"], label="validation loss")
ax1.legend()

ax2.plot(hist["train"],label="training accuracy")
ax2.plot(hist["val"],label="val accuracy")
ax2.legend()

plt.show()   

# %% [markdown]
# Training from scratch with only 100 examples per class does not allow the network to perform very well in this task. It reaches at most 70% accuracy, and the loss seems not descreassing anymore.

# %% [markdown]
# ## Preparing the model for finetuning and feature extraction
# 
# In order to load the the pretrained weights for the ResNet model we must change a bit the function `initialize model`. 

# %%
def initialize_model(num_classes):
    # Resnet18 with pretrained weights 
    model = models.resnet18(pretrained=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    
    model.fc = nn.Linear(512,num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model, input_size


# Number of classes in the dataset
num_classes = 2

# Initialize the model
model, input_size = initialize_model(num_classes)


# %% [markdown]
# Moreover, depending if we want to do finetuning (update all parameters) or if we want to do feature extraction (update only the last fully connected layer), we must specify wich parameter to update.
# 
# The following helper function sets the ``.requires_grad`` attribute of the parameters in the model. This is especially useful when you want to freeze part of your model, as the parameters with ``.requires_grad=False`` will not be updated during training.
# 

# %%
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# %% [markdown]
# CAUTION: you must call this function in the correct place, otherwise you will have no learnable parameters in your model.

# %% [markdown]
# ### Create the Optimizer
# 
# 
# The final step for finetuning and feature extracting is to create an optimizer that only updates the
# desired parameters. Recall that after loading the pretrained model, but before reshaping, if ``feature_extract=True`` we manually set all of the parameter’s ``.requires_grad`` attributes to False. Then the reinitialized layer’s parameters have ``.requires_grad=True`` by default. So now we know that *all parameters that have ``.requires_grad=True`` should be optimized. Next, we make a list of such parameters and input this list to the Adam algorithm constructor.
# 
# 

# %%
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

# %% [markdown]
# # HOMEWORK
# 

# %% [markdown]
# A) Train the model with feature extraction for 15 epochs. This is, you must freeze all parameters except the last fully connected layer. Plot the train/val losses and accuracies.

# %%
def initialize_model(num_classes):
    # Resnet18 with pretrained weights 
    model = models.resnet18(pretrained=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    set_parameter_requires_grad(model, True)
    model.fc = nn.Linear(512,num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model, input_size


# Number of classes in the dataset
num_classes = 2

# Initialize the model
model, input_size = initialize_model(num_classes)

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

# Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

# Train and evaluate
model2, hist2, losses2 = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# plot the losses and accuracies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(losses2["train"], label="training loss")
ax1.plot(losses2["val"], label="validation loss")
ax1.legend()

ax2.plot(hist2["train"],label="training accuracy")
ax2.plot(hist2["val"],label="val accuracy")
ax2.legend()

plt.show()   

# %% [markdown]
# B) Train the model finetuning all the parameters for 15 epochs. Plot the train/val losses and accuracies. 

# %%
def initialize_model(num_classes):
    # Resnet18 with pretrained weights 
    model = models.resnet18(pretrained=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    
    model.fc = nn.Linear(512,num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model, input_size


# Number of classes in the dataset
num_classes = 2

# Initialize the model
model, input_size = initialize_model(num_classes)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

# Train and evaluate
model3, hist3, losses3 = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# plot the losses and accuracies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(losses3["train"], label="training loss")
ax1.plot(losses3["val"], label="validation loss")
ax1.legend()

ax2.plot(hist3["train"],label="training accuracy")
ax2.plot(hist3["val"],label="val accuracy")
ax2.legend()

plt.show()   

# %% [markdown]
# 
# C) Plot the  train/val losses and accuracies of all three approaches: training from scratch, finetunning, and festure extraction. To make easier visualization and comparison, use four `plt.subplots`: one for training loss, one for val loss, one for training accuracy, and one for val accuracy.

# %%
# plot the losses and accuracies of all models
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 6))

ax1.plot(losses["train"], label="from scratch")
ax1.plot(losses2["train"], label="feature extraction")
ax1.plot(losses3["train"], label="finetunning")
ax1.set_title("training loss")
ax1.legend()

ax2.plot(losses["val"], label="from scratch")
ax2.plot(losses2["val"], label="feature extraction")
ax2.plot(losses3["val"], label="finetunning")
ax2.set_title("val loss")
ax2.legend()

ax3.plot(hist["train"],label="from scratch")
ax3.plot(hist2["train"],label="feature extraction")
ax3.plot(hist3["train"],label="finetunning")
ax3.set_title("training accuracy")
ax3.legend()

ax4.plot(hist["val"],label="from scratch")
ax4.plot(hist2["val"],label="feature extraction")
ax4.plot(hist3["val"],label="finetunning")
ax4.set_title("val accuracy")
ax4.legend()

# %% [markdown]
# D) Train the same model for a different dataset (MIT Scenes) using the three training strategies: from scratch, finetunning, and feature extraction. 
# 
# The URL of the dataset is the following:
# 
# https://xnap-datasets.s3.us-east-2.amazonaws.com/MIT_scenes.zip
# 
# The dataset contains scene images of 8 classes: coast, forest, highway, inside_city, mountain, open_country, street, and tallbuilding. The number of training images per class varies between 187 and 295. 
# 
# Train the three models and plot the  train/val losses and accuracies of all of them. Again, to make easier visualization and comparison, use four `plt.subplots`: one for training loss, one for val loss, one for training accuracy, and one for val accuracy.

# %%
# download the dataset
url = 'https://xnap-datasets.s3.us-east-2.amazonaws.com/MIT_scenes.zip'
#datasets.utils.download_and_extract_archive(url, './data')
datasets.utils.extract_archive('/home/alumne/extradisc/Datasets/MIT_split.zip', '/home/alumne/proves/data')

data_dir = "/home/alumne/proves/data/"

# Just normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(), # YOUR CODE HERE!
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}


# %%
def initialize_model(num_classes, pretrained=False, feature_extracting=False):
    # Resnet18 with pretrained weights 
    model = models.resnet18(pretrained=pretrained) 
    set_parameter_requires_grad(model, feature_extracting)
    model.fc = nn.Linear(512,num_classes)# YOUR CODE HERE!
    
    input_size = 224
        
    return model


# Number of classes in the dataset
num_classes = 8

# Initialize the model
all_models = [initialize_model(num_classes), initialize_model(num_classes, True), initialize_model(num_classes, True, True)]
model_labels = ['from scratch', 'fine_tunning', 'feature_extraction']

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 40

all_loss_hist = []
all_acc_hist = []

for model in all_models:

    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)

    # Send the model to GPU
    model = model.to(device)

    # Train and evaluate
    _, hist, losses = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    
    all_loss_hist.append(losses)
    all_acc_hist.append(hist)



# %%
# plot the losses and accuracies of all models
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 6))

ax1.plot(all_loss_hist[0]["train"], label="from scratch")
ax1.plot(all_loss_hist[2]["train"], label="feature extraction")
ax1.plot(all_loss_hist[1]["train"], label="finetunning")
ax1.set_title("training loss")
ax1.legend()

ax2.plot(all_loss_hist[0]["val"], label="from scratch")
ax2.plot(all_loss_hist[2]["val"], label="feature extraction")
ax2.plot(all_loss_hist[1]["val"], label="finetunning")
ax2.set_title("val loss")
ax2.legend()

ax3.plot(all_acc_hist[0]["train"],label="from scratch")
ax3.plot(all_acc_hist[2]["train"],label="feature extraction")
ax3.plot(all_acc_hist[1]["train"],label="finetunning")
ax3.set_title("training accuracy")
ax3.legend()

ax4.plot(all_acc_hist[0]["val"],label="from scratch")
ax4.plot(all_acc_hist[2]["val"],label="feature extraction")
ax4.plot(all_acc_hist[1]["val"],label="finetunning")
ax4.set_title("val accuracy")
ax4.legend()

# %%



