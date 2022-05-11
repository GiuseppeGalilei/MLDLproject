import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils


class GKTServerTrainer(object):
    def __init__(self, client_num, device, server_model, args):
        self.client_num = client_num
        self.device = device
        self.args = args

        # server model
        self.model_global = server_model
        self.model_global.to(self.device)

        self.model_global.train()

        self.optimizer = torch.optim.SGD(self.model_global.parameters(), lr=1e-2, momentum=0.5,weight_decay=0)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)
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

    def add_local_trained_result(self, index, extracted_feature_dict, logits_dict, labels_dict,
                                 extracted_feature_dict_test, labels_dict_test):

        self.client_extracted_feauture_dict[index] = extracted_feature_dict
        self.client_logits_dict[index] = logits_dict
        self.client_labels_dict[index] = labels_dict
        self.client_extracted_feauture_dict_test[index] = extracted_feature_dict_test
        self.client_labels_dict_test[index] = labels_dict_test


    def get_global_logits(self, client_index):
        return self.server_logits_dict[client_index]

    def train(self, round_idx):
        self.train_and_eval(round_idx, self.args.epochs_server)
        self.scheduler.step(self.best_acc)


    def train_and_eval(self, round_idx, epochs):
        for epoch in range(epochs):
            train_metrics = self.train_large_model_on_the_server()

            if epoch == epochs - 1:
                print({"train/loss": train_metrics['train_loss'],"train/accuracy": train_metrics['train_acc'], "epoch": round_idx + 1})

                # Evaluate for one epoch on validation set
                test_metrics = self.eval_large_model_on_the_server()

                # Find the best accTop1 model.
                test_acc = test_metrics['test_accTop1']

                print({"test/loss": test_metrics['test_loss'],"test/accuracy": test_metrics['test_acc'], "epoch": round_idx + 1})


    def train_large_model_on_the_server(self):
        # clear the server side logits
        for key in self.server_logits_dict.keys():
            self.server_logits_dict[key].clear()
        self.server_logits_dict.clear()

        self.model_global.train()

        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()

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

                loss_kd = self.criterion_KL(output_batch, batch_logits).to(self.device)
                loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                loss = loss_kd + self.args.alpha * loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update average loss and accuracy
                metrics = utils.accuracy(output_batch, batch_labels, topk=(1))
                accTop1_avg.update(metrics[0].item())
                loss_avg.update(loss.item())

                # update the logits for each client
                # Note that this must be running in the model.train() model,
                # since the client will continue the iteration based on the server logits.
                s_logits_dict[batch_index] = output_batch.cpu().detach().numpy()

        # compute mean of all metrics in summary
        train_metrics = {'train_loss': loss_avg.value(),
                         'train_acc': accTop1_avg.value()}

        return train_metrics

    def eval_large_model_on_the_server(self):

        # set model to evaluation mode
        self.model_global.eval()
        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()
        with torch.no_grad():
            for client_index in self.client_extracted_feauture_dict_test.keys():
                extracted_feature_dict = self.client_extracted_feauture_dict_test[client_index]
                labels_dict = self.client_labels_dict_test[client_index]

                for batch_index in extracted_feature_dict.keys():
                    batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                    batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                    output_batch = self.model_global(batch_feature_map_x)
                    loss = self.criterion_CE(output_batch, batch_labels)

                    # Update average loss and accuracy
                    metrics = utils.accuracy(output_batch, batch_labels, topk=(1))
                    # only one element tensors can be converted to Python scalars
                    accTop1_avg.update(metrics[0].item())
                    loss_avg.update(loss.item())

        # compute mean of all metrics in summary
        test_metrics = {'test_loss': loss_avg.value(),
                        'test_acc': accTop1_avg.value()}

        return test_metrics
