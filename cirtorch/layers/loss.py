import torch
import torch.nn as nn

import cirtorch.layers.functional as LF

# --------------------------------------
# Loss/Error layers
# --------------------------------------


class MyAggregateExponentialGammaLoss(nn.Module):

    r"""
    The loss from the paper:
        Miguel Molina-Moreno, Iván González-Díaz, Ralf Mikut, Fernando Díaz-de-María,
        A self-supervised embedding of cell migration features for behavior discovery over cell populations,
        Computer Methods and Programs in Biomedicine,
        2024,
        DOI:
    """

    def __init__(self, alpha=1.0, gamma=0.8, drop_loss=0, drop_loss_freq=0, eps=1e-6):
        super(MyAggregateExponentialGammaLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.drop_loss = drop_loss
        self.drop_loss_freq = drop_loss_freq
        self.eps = eps
        self.count = 0
        self.idx = None
        print('Creating myAggExpGammaLoss with gamma: {}, alpha: {}'.format(
            self.gamma, self.alpha))

    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):

        updateDropped = False
        if ((self.count % self.drop_loss_freq) == 0):
            updateDropped = True
            self.count = 0

        self.count += 1

        if (self.drop_loss > 0):
            if (updateDropped):
                numFeat = x.size()[0]
                self.idx = torch.rand(numFeat) > self.drop_loss

            x = x[self.idx, :]

        return LF.my_aggregate_exponential_gamma_loss(x, label, Lw, gamma=self.gamma,
                                                      alpha=self.alpha, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'


class MyAggregateExponentialGammaLossID(nn.Module):

    r"""
    The loss from the paper:
        Miguel Molina-Moreno, Iván González-Díaz, Ralf Mikut, Fernando Díaz-de-María,
        A self-supervised embedding of cell migration features for behavior discovery over cell populations,
        Computer Methods and Programs in Biomedicine,
        2024,
        DOI:
    """

    def __init__(self, alpha=1.0, gamma=0.8, drop_loss=0, drop_loss_freq=0, eps=1e-6):
        super(MyAggregateExponentialGammaLossID, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.drop_loss = drop_loss
        self.drop_loss_freq = drop_loss_freq
        self.eps = eps
        self.count = 0
        self.idx = None
        print('Creating myAggExpGammaLossID with gamma: {}, alpha: {}'.format(
            self.gamma, self.alpha))

    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):

        updateDropped = False
        if ((self.count % self.drop_loss_freq) == 0):
            updateDropped = True
            self.count = 0

        self.count += 1

        if (self.drop_loss > 0):
            if (updateDropped):
                numFeat = x.size()[0]
                self.idx = torch.rand(numFeat) > self.drop_loss

            x = x[self.idx, :]

        return LF.my_aggregate_exponential_gamma_loss_id(x, label, Lw, gamma=self.gamma,
                                                         alpha=self.alpha, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
