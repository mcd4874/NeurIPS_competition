import torch
# import matplotlib

# matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as col

from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from dassl.utils.meters import AverageMeter
import pandas as pd
import seaborn as sns
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               filename: str, source_color='r', target_color='b'):
#     """
#     Visualize features from different domains using t-SNE.
#     Args:
#         source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
#         target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
#         filename (str): the file name to save t-SNE
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)
#
#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
#
#     # domain labels, 1 represents source while 0 represents target
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#
#     # visualize using matplotlib
#     plt.figure(figsize=(10, 10))
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=2)
#     plt.savefig(filename)

def visualize(features,labels,domain_names,filename: str,source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (numpy): features from source domain in shape :math:`(minibatch, F)`
        target_feature (numpy): features from target domain in shape :math:`(minibatch, F)`
        source_label (numpy): label from source domain in shape : (minibatch,)
        target_label (numpy): label from target domain in shape : (minibatch,)
        source_domain (str): name of the source domain
        target_domain (str): name of target domain
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33,perplexity=50,init='pca').fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    # domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    table = pd.DataFrame({"dim_1":X_tsne[:, 0],
                  "dim_2":X_tsne[:, 1],
                  "classes":labels,
                  "domain":domain_names})


    # # visualize using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=2)
    # plt.savefig(filename)
    # a = sns.scatterplot(data=table,x='dim_1',y='dim_2',hue='domain',style='classes')
    a = sns.scatterplot(data=table,x='dim_1',y='dim_2',hue='classes')

    fig = a.get_figure()
    fig.savefig(filename)
    fig.clf()


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(feature: torch.Tensor, domain_label: torch.tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.
    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier
    Returns:
        :math:`\mathcal{A}`-distance
    """
    # source_label = torch.ones((source_feature.shape[0], 1))
    # target_label = torch.zeros((target_feature.shape[0], 1))
    # feature = torch.cat([source_feature, target_feature], dim=0)
    # label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, domain_label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter()
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc)
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct



import tqdm

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
            if max_num_features is not None and i >= max_num_features:
                break
    return torch.cat(all_features, dim=0)