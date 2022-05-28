import torch
from torch import nn
from torch.utils.data import DataLoader
import os

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
        
        pg = torch.cat([param.reshape(-1) for param in self.model.parameters()])

        for i in range(num_clients):
            torch.save({"prev_grads": pg},
                client_dir + f"{i}.pt")

        self.criterion = nn.CrossEntropyLoss()
        self.test_loader = DataLoader(testset, batch_size=100, 
            shuffle=False, num_workers=2)

        self.test_metrics_list = []

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

    def evaluate(self, round):
        self.model.eval()
        print("Evaluating server model at round", round, "...", end=" ")
        
        test_loss_avg = 0
        n = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, lbl in self.test_loader:
                img, lbl = img.to(self.device), lbl.to(self.device)
                y = self.model(img)
                n += 1
                test_loss_avg = (n-1) / n * test_loss_avg + 1 / n * self.criterion(y, lbl).item()
                _, predicted = torch.max(y.data, 1)
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()

        self.test_metrics_list.append({
            "round": round,
            "test_accuracy": correct / total,
            "test_avg_loss": test_loss
        })
        self.model.train()
        
        print("done!")

    def get_test_metrics(self):
        return self.test_metrics_list

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

        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(trainset, data_idxs), batch_size=128,
            num_workers=2, worker_init_fn=seed_worker, generator=g)

        self.train_metrics_list = []

    def train(self, model, server_state_dict, round):
        print("Training client", self.id, "...", end=" ")

        prev_grads = torch.load(client_dir + f"{self.id}.pt")["prev_grads"]
        model.load_state_dict(server_state_dict)
        model.to(self.device)
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.mm)
        model.train()

        loss_avg = 0
        n = 0
        metrics = []
        for epoch in range(self.local_epochs):
            for img, lbl in self.train_loader:
                optim.zero_grad()
                img, lbl = img.to(self.device), lbl.to(self.device)
                y = model(img)
                loss = self.criterion(y, lbl)
                n += 1
                loss_v = (n-1) / n * loss_avg + 1 / n * loss.item()

                cur_flat = torch.cat([p.reshape(-1) for p in model.parameters()])
                # Flatten the current server parameters
                par_flat = torch.cat([p.reshape(-1) for k, p in server_state_dict.items() if k in [k1 for k1, v in model.named_parameters()] ])
                #assert(cur_flat.requires_grad)

                lin_penalty = torch.sum(prev_grads * cur_flat)
                quad_penalty = self.alpha / 2 * torch.linalg.norm((cur_flat - par_flat), 2)**2

                loss = loss - lin_penalty + quad_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_value)
                optim.step()
            del img, lbl

        with torch.no_grad():  
            cur_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

            prev_grads -= self.alpha * (cur_flat - par_flat)
            torch.save({"prev_grads": prev_grads},
                client_dir + f"{self.id}.pt")
                    
        print(f"done! average batch loss={loss_avg}")
        return model.state_dict(), metrics
