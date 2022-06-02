import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from utils.reproducibility import seed_worker
from utils.datasets import DatasetSplit
from fedgkt.utils import *


g = torch.Generator()
clients_status_dir = "./feddyn/clients_status/"


class DYNServer():
    
    def __init__(self, model, alpha, tot_clients, testset, cuda, seed):
        g.manual_seed(seed)
        
        self.model = model
        self.alpha = alpha
        self.tot_clients = tot_clients
        self.cuda = cuda
        
        if self.cuda:
            self.model.cuda()
        
        self.testld = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        
        # compose server state
        self.h = dict()
        for key in self.model.state_dict():
            self.h[key] = torch.zeros_like(self.model.state_dict()[key])
            
        # generate the clients states
        if not os.path.exists(clients_status_dir):
            os.mkdir(clients_status_dir)
        for cid in range(self.tot_clients):
            torch.save({"prev_state": self.h}, clients_status_dir + f"{cid}.pt")
            
    def update(self, active_devices_models, com_round):
        print(f"Updating server model @ round {com_round}...", end=" ")
        
        num_participants = len(active_devices_models)
        
        self.h = {
            key: self.h[key] - self.alpha / self.tot_clients * sum(client_model[key] - self.model.state_dict()[key] for client_model in active_devices_models)
            for key in self.h
        }
        
        params = {
            key: sum(client_model[key] for client_model in active_clients_models) / num_participants - self.h[key] / self.alpha
            for key in self.h
        }
        
        self.model.load_state_dict(params)
        
        print("done!")
        
    def evaluate(self, com_round):
        print(f"Evaluating server model @ round {com_round}...", end=" ")
        
        criterion = nn.CrossEntropyLoss(reduction="sum")
        
        loss_value, correct, total = 0, 0, 0
        
        self.model.eval()
        for img, lbl in testld:
            if self.cuda:
                img, lbl = img.cuda(), lbl.cuda()
            lblhat = self.model(img)
            
            loss = criterion(lblhat, lbl)
            
            _, predicted = torch.max(lblhat.data, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()
        self.model.train()
        
        print(f"done!\t accuracy={(correct / total):.3} loss={(loss.item() / total):.3}")
        
    def get_state_dict(self):
        return self.model.state_dict()
    
    
class DYNClient():
    
    def __init__(self, cid, alpha, lr, weight_decay, momentum, clip_value, local_epochs, data_idxs, cuda):
        self.cid = cid
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.clip_value = clip_value
        self.local_epochs = local_epochs
        self.data_idxs = data_idxs
        self.cuda = cuda
        
    def train(self, model, trainset, server_state_dict, com_round):
        print(f"Training client {self.cid} @ round {com_round}...", end=" ")
        
        model.load_state_dict(server_state_dict)
        prev_status = torch.load(clients_status_dir + f"{self.cid}.pt")["prev_state"]
        trainld = DataLoader(DatasetSplit(trainset, self.data_idxs), batch_size=128, shuffle=True,
                             num_workers=2, worker_init_fn=seed_worker, generator=g)
        if self.cuda:
            model.cuda()
            
        optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        
        loss_value, correct, total = 0, 0, 0
        
        model.train()
        for epoch in range(self.local_epochs):
            for img, lbl in trainld:
                if self.cuda:
                    img, lbl = img.cuda(), lbl.cuda()

                optimizer.zero_grad()
                lblhat = model(img)
                loss = criterion(lblhat, lbl)

                lin_penalty, quad_penalty = 0, 0
                for key in model.state_dict():
                    lin_penalty += torch.sum(prev_status[key] * model.state_dict()[key])
                    quad_penalty += F.mse_loss(model.state_dict()[key].type(torch.float32), 
                                               server_state_dict[key].type(torch.float32), reduction="sum")
                print(f"loss={loss.item()}, lin_penalty={lin_penalty}, quad_penalty={quad_penalty}, ", end="")
                loss -= lin_penalty
                loss += self.alpha / 2 * quad_penalty
                print(f"modified_loss={loss.item()}")

                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(lblhat.data, 1)
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()
        model.eval()
                
        prev_status = {
            key: prev_status[key] - self.alpha * (model.state_dict()[key] - server_state_dict[key])
            for key in prev_status
        }
        torch.save(prev_status, clients_status_dir + f"{self.cid}.pt")
        
        print(f" done!\t avg_accuracy={(correct / total):.3} avg_loss={(loss.item() / total):.3}")
        
        return model.state_dict()
