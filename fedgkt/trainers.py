"""
Based on the implementation in https://github.com/FedML-AI/FedML
"""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.reproducibility import seed_worker, make_it_reproducible
from utils.datasets import DatasetSplit
from fedgkt.utils import *

g=torch.Generator()

class GKTServerTrainer(object):
    def __init__(self, client_num, device, server_model, args, seed):
        self.client_num = client_num
        self.device = device
        self.args = args

        self.model_global = server_model
        self.model_global.to(self.device)

        self.model_global.train()

        self.optimizer = optim.SGD(self.model_global.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(self.args['temperature'])
        self.best_acc = 0.0

        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feauture_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # key: client_index; value: labels_dict
        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feauture_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.train_metrics_list = []
        self.test_metrics_list = []

        g.manual_seed(seed)

    def add_local_trained_result(self, index, extracted_feature_dict, logits_dict, labels_dict,
                                 extracted_feature_dict_test, labels_dict_test):

        self.client_extracted_feauture_dict[index] = extracted_feature_dict
        self.client_logits_dict[index] = logits_dict
        self.client_labels_dict[index] = labels_dict
        self.client_extracted_feauture_dict_test[index] = extracted_feature_dict_test
        self.client_labels_dict_test[index] = labels_dict_test
        
        print(len(self.client_extracted_feauture_dict_test))

    def remove_records(self):
        for idx in self.client_extracted_feauture_dict.keys():
            self.client_extracted_feauture_dict[idx].clear()
            self.client_logits_dict[idx].clear()
            self.client_labels_dict[idx].clear()
            self.server_logits_dict[idx].clear()
        for id in self.client_extracted_feauture_dict_test.keys():
            self.client_extracted_feauture_dict_test[idx].clear()
            self.client_labels_dict_test[idx].clear()
        self.client_extracted_feauture_dict.clear()
        self.client_logits_dict.clear()
        self.client_labels_dict.clear()
        self.server_logits_dict.clear()
        self.client_extracted_feauture_dict_test.clear()
        self.client_labels_dict_test.clear()
        

    def get_global_logits(self, client_index):
        return self.server_logits_dict[client_index]

    def train(self, round_idx):
        self.train_and_eval(round_idx, self.args['epochs_server'])
        self.scheduler.step(self.best_acc)


    def train_and_eval(self, round_idx, epochs):
        for epoch in range(epochs):
            train_metrics = self.train_large_model_on_the_server(round_idx, epoch)
            self.train_metrics_list.append(train_metrics)
            print({"train/loss": train_metrics['train_loss'],"train/accuracy": train_metrics['train_acc'], "round": round_idx + 1})
            if epoch == epochs - 1:
                test_metrics = self.eval_large_model_on_the_server(round_idx)
                self.test_metrics_list.append(test_metrics)
                
                if test_metrics['test_acc'] >= self.best_acc:
                    self.best_acc= test_metrics['test_acc']
                
                print({"test/loss": test_metrics['test_loss'],"test/accuracy": test_metrics['test_acc'], "round": round_idx + 1})

    def train_large_model_on_the_server(self, round_idx, epoch):
        for key in self.server_logits_dict.keys():
            self.server_logits_dict[key].clear()
        self.server_logits_dict.clear()

        self.model_global.train()

        loss_avg = RunningAverage()
        accTop1_avg = RunningAverage()

        for client_index in self.client_extracted_feauture_dict.keys():
            extracted_feature_dict = self.client_extracted_feauture_dict[client_index]
            logits_dict = self.client_logits_dict[client_index]
            labels_dict = self.client_labels_dict[client_index]

            s_logits_dict = dict()
            self.server_logits_dict[client_index] = s_logits_dict
            for batch_index in extracted_feature_dict.keys():
                batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                batch_logits = torch.from_numpy(logits_dict[batch_index]).float().to(self.device)
                batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                output_batch = self.model_global(batch_feature_map_x)

#                 loss_kd = self.criterion_KL(output_batch, batch_logits).to(self.device)
                loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
#                 loss = loss_kd + self.args['alpha'] * loss_true
                loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics = accuracy(output_batch, batch_labels, topk=(1,))
                accTop1_avg.update(metrics[0].item())
                loss_avg.update(loss.item())

                s_logits_dict[batch_index] = output_batch.cpu().detach().numpy()

        train_metrics = {
            "round": round_idx,
            "epoch": epoch,
            'train_loss': loss_avg.value(),
            'train_acc': accTop1_avg.value()}

        return train_metrics

    def eval_large_model_on_the_server(self, round_idx):
        self.model_global.eval()
        loss_avg = RunningAverage()
        accTop1_avg = RunningAverage()
        with torch.no_grad():
            for client_index in self.client_extracted_feauture_dict_test.keys():
                extracted_feature_dict = self.client_extracted_feauture_dict_test[client_index]
                labels_dict = self.client_labels_dict_test[client_index]

                for batch_index in extracted_feature_dict.keys():
                    batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                    batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                    output_batch = self.model_global(batch_feature_map_x)
                    loss = self.criterion_CE(output_batch, batch_labels)

                    metrics = accuracy(output_batch, batch_labels, topk=(1,))
                    accTop1_avg.update(metrics[0].item())
                    loss_avg.update(loss.item())

        test_metrics = {
            "round": round_idx,
            'test_loss': loss_avg.value(),
            'test_acc': accTop1_avg.value()}

        return test_metrics

    def get_metrics_lists(self):
        return self.train_metrics_list, self.test_metrics_list


class GKTClientTrainer(object):
    def __init__(self, client_index, local_training_data, local_test_data, local_sample_number, device,
                 client_model, args):
        self.client_index = client_index

        self.local_training_data = local_training_data
        self.local_test_data = local_test_data

        self.local_sample_number = local_sample_number

        self.args = args

        self.device = device
        self.client_model = client_model

        self.client_model.to(self.device)

        self.optimizer = optim.SGD(self.client_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(self.args['temperature'])

        self.server_logits_dict = dict()

    def get_sample_number(self):
        return self.local_sample_number

    def update_large_model_logits(self, logits):
        self.server_logits_dict = logits

    def train(self):
        # key: batch_index; value: extracted_feature_map
        extracted_feature_dict = dict()

        # key: batch_index; value: logits
        logits_dict = dict()

        # key: batch_index; value: label
        labels_dict = dict()

        # for test - key: batch_index; value: extracted_feature_map
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        self.client_model.train()
        epoch_loss = []
        for epoch in range(self.args['epochs_client']):
            batch_loss = []
            trainloader = DataLoader(DatasetSplit(self.local_training_data, self.local_sample_number), batch_size = 128, shuffle=True, num_workers=2, worker_init_fn = seed_worker, generator=g)
            for batch_idx, data in enumerate(trainloader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs, _ = self.client_model(images)
                loss_true = self.criterion_CE(log_probs, labels)
                if len(self.server_logits_dict) != 0:
                    large_model_logits = torch.from_numpy(self.server_logits_dict[batch_idx]).to(
                        self.device)
                    loss_kd = self.criterion_KL(log_probs, large_model_logits)
                    loss = loss_true + self.args['alpha'] * loss_kd
                else:
                    loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('client {} - Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.client_index, epoch, batch_idx * len(images), len(trainloader),
                                              100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.client_model.eval()

        for batch_idx, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            log_probs, extracted_features = self.client_model(images)

            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()

        testloader = DataLoader(self.local_test_data, batch_size = 128, shuffle=True, num_workers=2, worker_init_fn = seed_worker, generator=g)
        for batch_idx, data in enumerate(testloader):
            images, labels = data
            test_images, test_labels = images.to(self.device), labels.to(self.device)
            _, extracted_features_test = self.client_model(test_images)
            extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
            labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test
