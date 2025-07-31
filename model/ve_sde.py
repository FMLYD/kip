import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

class SBGM(nn.Module):
    def __init__(self, snet, sigma, D, T):
        super(SBGM, self).__init__()

        # sigma parameter
        self.sigma = torch.Tensor([sigma])

        # define the base distribution (multivariate Gaussian with the diagonal covariance)
        var = (1. / (2. * torch.log(self.sigma))) * (self.sigma ** 2 - 1.)
        self.base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(D), var * torch.eye(D))

        # score model
        self.snet = snet

        # time embedding (a single linear layer)
        self.time_embedding = nn.Sequential(nn.Linear(1, D), nn.Tanh())

        # other hyperparams
        self.D = D

        self.T = T

        self.EPS = 1.e-5

    def sigma_fun(self, t):
        # the sigma function (dependent on t), it is the std of the distribution
        return torch.sqrt((1. / (2. * torch.log(self.sigma))) * (self.sigma ** (2. * t) - 1.))

    def log_p_base(self, x):
        # the log-probability of the base distribition, p_1(x)
        log_p = self.base.log_prob(x)
        return log_p

    def sample_base(self, x_0):
        # sampling from the base distribution
        return self.base.rsample(sample_shape=torch.Size([x_0.shape[0]]))

    def sample_p_t(self, x_0, x_1, t):
        # sampling from p_0t(x_t|x_0)
        # x_0 ~ data, x_1 ~ noise
        x = x_0 + self.sigma_fun(t) * x_1

        return x

    def lambda_t(self, t):
        # the loss weighting
        return self.sigma_fun(t) ** 2

    def diffusion_coeff(self, t):
        # the diffusion coefficient in the SDE
        return self.sigma ** t

    def forward(self, x_0, reduction='mean'):
        # =====
        # x_1 ~ the base distribiution
        x_1 = torch.randn_like(x_0)
        # t ~ Uniform(0, 1)
        t = torch.rand(size=(x_0.shape[0], 1)) * (1. - self.EPS) + self.EPS

        # =====
        # sample from p_0t(x|x_0)
        x_t = self.sample_p_t(x_0, x_1, t)

        # =====
        # invert noise
        # NOTE: here we use the correspondence eps_theta(x,t) = -sigma*t score_theta(x,t)
        t_embd = self.time_embedding(t)
        x_pred = -self.sigma_fun(t) * self.snet(x_t + t_embd)

        # =====LOSS: Score Matching
        # NOTE: since x_pred is the predicted noise, and x_1 is noise, this corresponds to Noise Matching
        #       (i.e., the loss used in diffusion-based models by Ho et al.)
        SM_loss = 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)

        if reduction == 'sum':
            loss = SM_loss.sum()
        else:
            loss = SM_loss.mean()

        return loss

    def sample(self, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        x_t = self.sample_base(torch.empty(batch_size, self.D))

        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign!
        ts = torch.linspace(1., self.EPS, self.T)
        delta_t = ts[0] - ts[1]

        for t in ts[1:]:
            tt = torch.Tensor([t])
            u = 0.5 * self.diffusion_coeff(tt) * self.snet(x_t + self.time_embedding(tt))
            x_t = x_t - delta_t * u

        x_t = torch.tanh(x_t)
        return x_t

    def log_prob_proxy(self, x_0, reduction="mean"):
        # Calculate the proxy of the log-likelihood (see (Song et al., 2021))
        # NOTE: Here, we use a single sample per time step (this is done only for simplicity and speed);
        # To get a better estimate, we should sample more noise
        ts = torch.linspace(self.EPS, 1., self.T)

        for t in ts:
            # Sample noise
            x_1 = torch.randn_like(x_0)
            # Sample from p_0t(x_t|x_0)
            x_t = self.sample_p_t(x_0, x_1, t)
            # Predict noise
            t_embd = self.time_embedding(torch.Tensor([t]))
            x_pred = -self.snet(x_t + t_embd) * self.sigma_fun(t)
            # loss (proxy)
            if t == self.EPS:
                proxy = 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)
            else:
                proxy = proxy + 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)

        if reduction == "mean":
            return proxy.mean()
        elif reduction == "sum":
            return proxy.sum()


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.log_prob_proxy(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name='', T=None):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    if T is not None:
        model_best.T = T

    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('proxy')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val

# transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)  # changing to [-1, 1]


class VESDEImputation(nn.Module):
    def __init__(self, data_dim, hidden_dim, layer_num, lr, T=20, sigma=1.01, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(VESDEImputation, self).__init__()

        # define the score network
        self.score_net = nn.Sequential()
        for layer in range(layer_num):
            if layer == 0:
                self.score_net.append(nn.Linear(data_dim, hidden_dim))
            else:
                self.score_net.append(nn.Linear(hidden_dim, hidden_dim))
            if layer != (layer_num):
                self.score_net.append(nn.SiLU())
            else:
                self.score_net.append(nn.Hardtanh(min_val=-3., max_val=3.))

        self.model = SBGM(self.score_net, sigma=sigma, D=data_dim, T=T)

        self.loss_list = []
        self.optimizer = torch.optim.Adamax([p for p in self.score_net.parameters() if p.requires_grad==True], lr=lr)
        self.device = device

    def train_ve_sde(self, data_loader, epoch):
        for e in range(epoch):
            self.model.train()
            for idx, batch in enumerate(data_loader):
                loss = self.model.forward(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluation(self, data_loader):

        return None

    def evaluation(self, test_loader, name=None, model_best=None, epoch=None):
        # EVALUATION
        if model_best is None:
            # load best performing model
            model_best = torch.load(name + '.model')

        model_best.eval()
        loss = 0.
        N = 0.
        for indx_batch, test_batch in enumerate(test_loader):
            loss_t = model_best.log_prob_proxy(test_batch, reduction='sum')
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
        loss = loss / N

        if epoch is None:
            print(f'FINAL LOSS: nll={loss}')
        else:
            print(f'Epoch: {epoch}, val nll={loss}')

        return loss

    def samples_real(self, name, test_loader):
        # REAL-------
        num_x = 4
        num_y = 4
        x = next(iter(test_loader)).detach().numpy()

        fig, ax = plt.subplots(num_x, num_y)
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(x[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.axis('off')

        plt.savefig(name + '_real_images.pdf', bbox_inches='tight')
        plt.close()
        return None

    def samples_generated(self, name, data_loader, extra_name='', T=None):
        # GENERATIONS-------
        model_best = torch.load(name + '.model')
        model_best.eval()

        if T is not None:
            model_best.T = T

        num_x = 4
        num_y = 4
        x = model_best.sample(batch_size=num_x * num_y)
        x = x.detach().numpy()

        fig, ax = plt.subplots(num_x, num_y)
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(x[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.axis('off')

        plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
        plt.close()
        return None

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(self.device)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(self.device)
        mask = mask.to(self.device)
        imps.requires_grad = False

