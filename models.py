import torch
from torch import nn
from layers import SparseNGCNLayer, DenseNGCNLayer
from torch.nn import functional as F

class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.dropout = self.args.dropout
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i - 1], i, self.args.dropout, self.args.device) for i
                             in range(1, self.order_1 + 1)]
        self.upper_layers = nn.ModuleList(self.upper_layers)

        self.bottom_layers = [
            DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i - 1], i, self.args.dropout, self.args.device) for i in
            range(1, self.order_2 + 1)]
        self.bottom_layers = nn.ModuleList(self.bottom_layers)

        # self.combine_feat = nn.Linear(self.abstract_feature_number_2, self.abstract_feature_number_2)
        self.bilinear = nn.Bilinear(self.abstract_feature_number_2, self.abstract_feature_number_2, self.args.hidden1)
        self.decoder = nn.Sequential(nn.Linear(self.args.hidden1, self.args.hidden2),
                                     nn.ELU(),
                                     nn.Linear(self.args.hidden2, 1)
                                     )

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def embed(self, normalized_adjacency_matrix, features):
        """
                Forward pass.
                :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
                :param features: Feature matrix.
                :return predictions: Label predictions.
                """
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_1 = F.dropout(abstract_features_1, self.dropout, training=self.training)

        abstract_features_2 = torch.cat(
            [self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],
            dim=1)
        feat = F.dropout(abstract_features_2, self.dropout, training=self.training)

        # feat = F.elu(self.combine_feat(feat))

        return feat

    def forward(self, normalized_adjacency_matrix, features, idx):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        latent_features = self.embed(normalized_adjacency_matrix, features)
        
        feat_p1 = latent_features[idx[0]]
        feat_p2 = latent_features[idx[1]]
        feat = F.elu(self.bilinear(feat_p1, feat_p2))
        feat = F.dropout(feat, self.dropout, training=self.training)
        predictions = self.decoder(feat)
        return predictions, latent_features
