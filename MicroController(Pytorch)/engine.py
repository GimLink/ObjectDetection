from config import DEVICE, NUM_CLASSES,NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from utils import Averager
from model import create_model
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

# function for running training iterations
def train(train_data_loader, model, optimizer):
    print('Training...')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total = len(train_data_loader))

    for i , data in enumerate(prog_bar) :
        optimizer.zero_grad()
        images, target = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k ,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1
        # update the loss value beside the [rpgress bar for each iteration
        prog_bar.set_description(desc=f'Loss : {loss_value:.4f}')

    return train_loss_list

def validate(valid_data_loader, model) :
    print('Validating...')
    global val_itr
    global val_loss_list

    prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))

    for i , data in enumerate(prog_bar) :
        images, target = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k ,v in t.item()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            val_loss_hist.send(loss_value)

            val_itr += 1

            prog_bar.set_description(desc = f'Loss : {loss_value:.4f}')

    return val_loss_list

# maoin code

if __name__ == '__main__':
    model = create_model(num_classes = NUM_CLASSES)
    model.to(DEVICE)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []

    # name to save the trainde model with
    MODEL_NAME = 'model'

    if VISUALIZE_TRANSFORMED_IMAGES :
        from utils import show_transformed_image
        show_transformed_image(train_loader)

    # start the training epochs
    for epoch in range(NUM_EPOCHS) :
        print(f'\nEpoch {epoch +1} of {NUM_EPOCHS}')

        train_loss_hist.rest()
        val_loss_hist.rest()

        #create two subplots one for each train and val
        figure_1, train_ax = plt.subplots()
        figure_2, val_ax = plt.subplots()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f'Epoch # {epoch} train loss : {train_loss_hist.value:.3f}')
        print(f'Epoch # {epoch} validation loss : {val_loss_hist.value:.3f}')
        end = time.time()
        print(f'Took {((end - start)/60):/3f} minutes for epoch # {epoch}')

