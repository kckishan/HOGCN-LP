import torch
import numpy as np
import pandas as pd
from torch.utils import data
from gat_utils import load_data
from GAT import SPGAT
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os
import time


class Trainer(object):
    """
    Class for training the neural network.
    :param args: Arguments object.
    """

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.setup_features()
        self.model = SPGAT(self.args, self.feature_number, 1)
        self.model = self.model.to(self.device)
        print(self.model)

    def setup_features(self):
        """
        Creating a feature matrix, target vector and propagation matrix.
        """
        # Load data
        self.propagation_matrix, self.features, self.idx_map, Data_class = load_data(self.args)
        self.propagation_matrix = self.propagation_matrix.to(self.device)
        self.features = self.features.to(self.device)
        train_params = {'batch_size': self.args.batch_size,
                        'shuffle': True,
                        'num_workers': 6,
                        'drop_last': True}

        test_params = {'batch_size': self.args.batch_size,
                       'shuffle': False,
                       'num_workers': 6}

        data_path = f"./data/{self.args.network_type}/fold{self.args.fold_id}"
        if self.args.ratio:
            data_path = f"./data/{self.args.network_type}/{self.args.train_percent}/fold{self.args.fold_id}"

        print(f"Data folder: {data_path}")
        df_train = pd.read_csv(data_path + '/train.csv')
        df_val = pd.read_csv(data_path + '/val.csv')
        df_test = pd.read_csv(data_path + '/test.csv')

        training_set = Data_class(self.idx_map, df_train.label.values, df_train)
        self.train_loader = data.DataLoader(training_set, **train_params)

        validation_set = Data_class(self.idx_map, df_val.label.values, df_val)
        self.val_loader = data.DataLoader(validation_set, **test_params)

        test_set = Data_class(self.idx_map, df_test.label.values, df_test)
        self.test_loader = data.DataLoader(test_set, **test_params)

        self.feature_number = self.features.shape[1]

        # saving the results
        if self.args.ratio:
            self.model_save_folder = f"trained_models/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/fold{self.args.fold_id}/"
        else:
            self.model_save_folder = f"trained_models/network_{self.args.network_type}/order_{len(self.args.layers_1)}/fold{self.args.fold_id}/"

        if not os.path.exists(self.model_save_folder):
            os.makedirs(self.model_save_folder)

    def fit(self):
        """
        Fitting a neural network with early stopping.
        """
        no_improvement = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        max_auc = 0
        loss_history = []

        t_total = time.time()
        print('Start Training...')
        for epoch in range(self.args.epochs):
            t = time.time()
            print('-------- Epoch ' + str(epoch + 1) + ' --------')
            y_pred_train = []
            y_label_train = []

            epoch_loss = 0
            for i, (label, pairs) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction, latent_feat = self.model(self.propagation_matrix, self.features, pairs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction.squeeze(), label.float())

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                loss_history.append(loss)

                label_ids = label.to('cpu').numpy()
                y_label_train = y_label_train + label_ids.flatten().tolist()
                y_pred_train = y_pred_train + prediction.flatten().tolist()

                if i % 100 == 0:
                    print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                        loss.cpu().detach().numpy()))

            roc_train = roc_auc_score(y_label_train, y_pred_train)

            # validation after each epoch
            if not self.args.fastmode:
                preds, roc_val, prc_val, f1_val, loss_val = self.score(self.val_loader)
                if roc_val > max_auc:
                    max_auc = roc_val
                    torch.save(self.model, f"{self.model_save_folder}model_{self.args.network_type}.pt")
                    no_improvement = 0
                else:
                    no_improvement = no_improvement + 1
                    if no_improvement == self.args.early_stopping:
                        break

                print('epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss.item()),
                      'auroc_train: {:.4f}'.format(roc_train),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'auroc_val: {:.4f}'.format(roc_val),
                      'auprc_val: {:.4f}'.format(prc_val),
                      'f1_val: {:.4f}'.format(f1_val),
                      'time: {:.4f}s'.format(time.time() - t))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.model = torch.load(f"{self.model_save_folder}model_{self.args.network_type}.pt")
        prediction, auroc_test, prc_test, f1_test, loss_test = self.score(self.test_loader)
        print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
              'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

        # saving the results
        results = {"prediction": prediction, "auroc": auroc_test, "pr": prc_test, "f1": f1_test}
        if self.args.ratio:
            save_folder = f"results/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/"
        else:
            save_folder = f"results/network_{self.args.network_type}/order_{len(self.args.layers_1)}/"

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_name = f"{save_folder}input_{self.args.input_type}_fold{self.args.fold_id}_lr{self.args.learning_rate}" \
                    f"_bs{self.args.batch_size}_hidden1_{self.args.hidden1}_hidden2_{self.args.hidden2}_dropout{self.args.dropout}.pt"
        torch.save(results, file_name)

        # saving embeddings
        latent_features = self.model.embed(self.features, self.propagation_matrix)
        embeddings = {"idxmap": self.idx_map, "emb": latent_features}

        if self.args.ratio:
            emb_folder = f"embeddings/network_{self.args.network_type}/{self.args.train_percent}/order_{len(self.args.layers_1)}/"
        else:
            emb_folder = f"embeddings/network_{self.args.network_type}/order_{len(self.args.layers_1)}/"

        if not os.path.exists(emb_folder):
            os.makedirs(emb_folder)

        file_name = f"{emb_folder}input_{self.args.input_type}_fold{self.args.fold_id}_lr{self.args.learning_rate}" \
                    f"_bs{self.args.batch_size}_hidden1_{self.args.hidden1}_hidden2_{self.args.hidden2}_dropout{self.args.dropout}.pt"
        torch.save(embeddings, file_name)

    def score(self, data_loader):
        """
        Scoring a neural network.
        :param indices: Indices of nodes involved in accuracy calculation.
        :return predictions: Probability for link existence
                roc_score: Area under ROC curve
                pr_score: Area under PR curve
                f1_score: F1 score
        """
        self.model.eval()
        y_pred = []
        y_label = []

        for i, (label, pairs) in enumerate(data_loader):
            label = label.to(self.device)
            output, latent_feat = self.model(self.propagation_matrix, self.features, pairs)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(), label.float())

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        return y_pred, roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                          outputs), loss
