import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from torch.nn.functional import mse_loss

from utils.reproducibility import seed_worker, make_it_reproducible
from utils.datasets import DatasetSplit

g = torch.Generator()

client_dir = "./feddyn/client_states/"


class FedDynServer():
    def __init__(self, model, alpha, num_clients, device, testset, seed):
        g.manual_seed(seed)
        
        self.num_clients = num_clients
        self.alpha = alpha
        self.device = device
        self.model = model.to(self.device)
        self.seed = seed

        self.h = dict()
        for key in self.model.state_dict():
            self.h[key] = torch.zeros_like(self.model.state_dict()[key])

        if not os.path.exists(client_dir):
            os.mkdir(client_dir)

        for i in range(num_clients):
            torch.save({"prev_grads": self.h},
                client_dir + f"{i}.pt")

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.test_loader = DataLoader(testset, batch_size=100, 
            shuffle=False, num_workers=2)
    
    def get_model_params_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_params_norm = total_norm ** 0.5
        return total_params_norm

    def update_model(self, active_clients_states):
        print("Updating server model...", end=" ")
        num_participants = len(active_clients_states)

        self.h = {
            key: h - self.alpha / self.num_clients * sum(theta[key] - server_param for theta in active_clients_states)
            for (key, h), server_param in zip(self.h.items(), self.model.state_dict().values())
        }
        
        par = {
            key: 1 / num_participants * sum(theta[key] for theta in active_clients_states)
            for key in self.model.state_dict().keys()
        }
        
        par = {
            key: param - 1 / self.alpha * h_param
            for (key, param), h_param in zip(par.items(), self.h.values())
        }
            
        self.model.load_state_dict(par)
        print("done!")

        hasnan = False
        for key in self.model.state_dict():
            if torch.isnan(self.model.state_dict()[key]).any():
                print(f"\t CLient model: {key} has some nan.")
                hasnan = True
        if not hasnan:
            print("\t No nan found in server model!")
        
        print(f"\t Server model norm: {self.get_model_params_norm()}")

    def evaluate(self, round):
        self.model.eval()
        print("Evaluating server model at round", round, "...", end=" ")
        
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

        metrics = {
            "round": round,
            "test_accuracy": correct / total,
            "test_avg_loss_per_image": test_loss / total
        }
        self.model.train()
        
        print("done!")
        return metrics

    def get_server_state(self):
        return self.model.state_dict()


class FedDynClient():
    def __init__(self, device, lr, wd, mm, alpha, id, local_epochs, trainset, data_idxs, clip_value):
        self.device = device
        self.id = id
        self.alpha = alpha
        self.lr = lr
        self.wd = wd
        self.mm = mm
        self.local_epochs = local_epochs
        self.clip_value = clip_value

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.train_loader = DataLoader(DatasetSplit(trainset, data_idxs), batch_size=128,
            num_workers=2, worker_init_fn=seed_worker, generator=g)

    def train(self, model, server_state_dict, round):
        print("Training client", self.id, "...")

        prev_grads = torch.load(client_dir + f"{self.id}.pt")["prev_grads"]
        model.load_state_dict(server_state_dict)
        model.to(self.device)
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.mm)
        model.train()

        loss_value = 0
        correct = 0
        total = 0
        for epoch in range(self.local_epochs):
            for img, lbl in self.train_loader:
                optim.zero_grad()
                img, lbl = img.to(self.device), lbl.to(self.device)
                y = model(img)
                loss = self.criterion(y, lbl)
                
                _, predicted = torch.max(y.data, 1)
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()

                lin_penalty = 0
                quad_penalty = 0
                for key in model.state_dict().keys():
                     lin_penalty += torch.sum(prev_grads[key] * model.state_dict()[key])
                     quad_penalty += mse_loss(model.state_dict()[key].type(torch.DoubleTensor), server_state_dict[key].type(torch.DoubleTensor), reduction='sum')
                
                print(f"\t loss:{loss.item()}, lin:{lin_penalty}, quad:{quad_penalty}", end="")
                    
                loss -= lin_penalty
                loss += self.alpha / 2. * quad_penalty
                loss.backward()
                print(f", modified loss:{loss.item()}")
                
                loss_value += loss.item()
                
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_value)
                optim.step()
            del img, lbl

        with torch.no_grad():
            for key in model.state_dict():
                if prev_grads[key].dtype != torch.float32:
                    prev_grads[key] -= (self.alpha * (model.state_dict()[key] - server_state_dict[key])).long()
                else:
                    prev_grads[key] -= self.alpha * (model.state_dict()[key] - server_state_dict[key])
            torch.save({"prev_grads": prev_grads},
                client_dir + f"{self.id}.pt")
            
        total_norm = 0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_params_norm = total_norm ** 0.5
        
        print("\t Client model norm: ", total_params_norm)

        hasnan = False
        for key in prev_grads:
            if torch.isnan(prev_grads[key]).any():
                print(f"\t Client state: {key} has some nan.")
                hasnan = True
        if not hasnan:
            print("\t No nan found in client state!")

        hasnan = False
        for key in model.state_dict():
            if torch.isnan(model.state_dict()[key]).any():
                print(f"\t Client model: {key} has some nan.")
                hasnan = True
        if not hasnan:
            print("\t No nan found in client model!")
    
        metrics = {
            "round": round,
            "train_avg_loss_per_image": loss_value / total,
            "train_accuracy": correct / total
        }
                    
        print(f"done! train loss per image={loss_value / total}\t train accuracy={correct / total}")
        return model.state_dict(), metrics
