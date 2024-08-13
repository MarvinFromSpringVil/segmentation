# -*- coding: utf-8 -*-


from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
from data_loader import get_dataloader
from unet_model import Unet
from utils import eval_print_metrics
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


def model_train(dataloader, device, net, epochs=500, batch_size=2, lr=1e-2, save_every=5, eval_every=10, is_eval=True):
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #loss_func = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    net.train()
 
    for epoch in range(1, epochs+1):
        epoch_tot_loss = 0
        print("[+] ====== Start training... epoch {} ======".format(epoch + 1))
                
        running_loss = 0.0

        for x, y, z in tqdm(dataloader):
            x = x.to(device) # image 
            y = y.to(device) # mask 
            z = z.to(device) # one-hot vector 

            
            pred = net(x)
                        
            loss = criterion(pred, y.float())
            running_loss += loss.item()
            print("[*] Epoch: {}, current loss: {:.4f}".format(epoch, loss.item()))

            optimizer.zero_grad()      
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            torch.save(net.state_dict(), "./log/weight_EP-{}_loss-{:.4f}.pth".format(epoch, running_loss))
    return net

if __name__ == "__main__":
    os.makedirs('./log', exist_ok=True)
    


    DATASET_ROOT = './pathology_dataset'
    TRRANSFORM = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL images to tensors
    ])

    # set parameters for training
    LR = 0.1
    EPOCHS = 100
    BATCH_SIZE = 10
    SAVE_EVERY = 20
    EVAL_EVERY = 30

    dataloader = get_dataloader(DATASET_ROOT, BATCH_SIZE, TRRANSFORM)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_ins = Unet(img_ch=3, isDeconv=True, isBN=True)
    unet_ins.to(device)
    trained_unet = model_train(dataloader, device,
                            unet_ins, 
                            batch_size=BATCH_SIZE, 
                            lr=LR, 
                            epochs=EPOCHS, 
                            save_every=SAVE_EVERY, 
                            eval_every=EVAL_EVERY)
