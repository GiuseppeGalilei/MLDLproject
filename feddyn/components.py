from numpy import isin
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
from collections import defaultdict

from utils.reproducibility import seed_worker, make_it_reproducible
from utils.datasets import DatasetSplit

g = torch.Generator()

class FedDynServer():
    def __init__(self, model, alpha, num_clients, device, testset, seed):
        make_it_reproducible(seed)
        g.manual_seed(seed)
        
        self.num_clients = num_clients
        self.alpha = alpha
        self.device = device
        self.model = model.to(self.device)
        self.seed = seed

        self.h = copy.deepcopy(self.model.state_dict())

        self.criterion = nn.CrossEntropyLoss()
        self.test_loader = DataLoader(testset, batch_size=100, 
            shuffle=False, num_workers=2)

        self.test_metrics_list = []

    def update_model(self, active_clients_states):
        print("Updating server model...", end=" ")
        num_participants = len(active_clients_states)

        sum_deltas = defaultdict(lambda: 0.0)
        for client_state in active_clients_states:
            for key in client_state.keys():
                sum_deltas[key] += client_state[key] - self.model.state_dict()[key]
        
        # update h
        for key in self.h.keys():
            self.h[key] -= (self.alpha * sum_deltas[key] / self.num_clients).type(self.h[key].dtype)

        sum_thetas = defaultdict(lambda: 0.0)
        for client_state in active_clients_states:
            for key in client_state.keys():
                sum_thetas[key] += client_state[key]

        # update server model
        for key in self.model.state_dict().keys():
            self.model.state_dict()[key] = (sum_thetas[key] / num_participants - self.h[key] / self.alpha).type(self.model.state_dict()[key].dtype)
            
        print("done!")

    def evaluate(self, round):
        self.model.eval()
        print("Evaluating model at round", round, "...", end=" ")
        
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, lbl in self.test_loader:
                img, lbl = img.to(self.device), lbl.to(self.device)
                y = self.model(img)
                test_loss += self.criterion(y, lbl).item()
                _, predicted = torch.max(y.data, 1)
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()

        self.test_metrics_list.append({
            "round": round,
            "test_accuracy": correct / total,
            "test_avg_loss": test_loss / len(self.test_loader)
        })
        
        print("done!")

    def get_test_metrics(self):
        return self.test_metrics_list

    def get_server_state(self):
        return self.model.state_dict()


class FedDynClient():
    def __init__(self, model, device, lr, alpha, id, local_epochs, trainset, data_idxs):
        self.device = device
        self.model = model.to(self.device)
        self.id = id
        self.alpha = alpha
        self.lr = lr
        self.local_epochs = local_epochs
        self.trainset = trainset
        self.data_idxs = data_idxs

        self.criterion = nn.CrossEntropyLoss()
        self.optim = None
        self.train_loader = DataLoader(DatasetSplit(self.trainset, self.data_idxs), batch_size=128,
            num_workers=2, worker_init_fn=seed_worker, generator=g)

        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

        self.train_metrics_list = []

    def train(self, server_state_dict, round):
        self.model.load_state_dict(server_state_dict)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        self.model.train()
        print("Training client", self.id, "...", end=" ")

        loss_v = 0
        for epoch in range(self.local_epochs):
            for img, lbl in self.train_loader:
                self.optim.zero_grad()
                img, lbl = img.to(self.device), lbl.to(self.device)
                y = self.model(img)
                loss = self.criterion(y, lbl)
                loss_v += loss.item()

                cur_params = None
                for param in self.model.parameters():
                    if not isinstance(cur_params, torch.Tensor):
                        cur_params = param.view(-1)
                    else:
                        cur_params = torch.cat((cur_params, param.view(-1)), dim=0)
                lin_penalty = torch.sum(cur_params * self.prev_grads)

                ser_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(ser_params, torch.Tensor):
                        ser_params = server_state_dict[name].view(-1)
                    else:
                        ser_params = torch.cat((ser_params, server_state_dict[name].view(-1)), dim=0)
                quad_penalty = self.alpha / 2 * torch.linalg.norm((cur_params - ser_params), 2)**2

                mod_loss = loss - lin_penalty + quad_penalty
                mod_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optim.step()
                
        delta_param = None
        for name, param in self.model.named_parameters():
            if not isinstance(delta_param, torch.Tensor):
                delta_param = param.view(-1) - server_state_dict[name].view(-1)
            else:
                delta_param  = torch.cat((delta_param, (param.view(-1) - server_state_dict[name].view(-1))), dim=0)

        self.prev_grads -= self.alpha * delta_param
                    
        print("done!")
        return self.model.state_dict(), metrics

