import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from utils.models import HighwayHuman, BayesOptWrapper, MLPHuman, ReplayMemory, LIMIT
import os, pickle


def save_data(data, path: str):
    """
    `data`: the data to serialize
    `path`: path to file to save
    """
    dir = path[0 : path.rindex("/")]
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    return


def train_bayes(rew: float, model: BayesOptWrapper):
    model.optimizer.register(params=model.current_matrix_params, target=rew)
    model.current_matrix_params = model.optimizer.suggest(model.utility)
    return


def train_LIMIT(
    model: LIMIT,
    memory: ReplayMemory,
    human_optim: Adam,
    decoder_optim: Adam,
    epochs=16,
    batch_size=256,
    prior_weight=1.0,
    non_prior_weight=1.0,
    prior_training_func=None,
    timesteps=1,
):
    """
    `model`: instance of LIMIT
    `memory`: ReplayMemory buffer
    `prior_training_func`: training function to call if training for priors is
                           desired, `None` otherwise. prior training func accepts arguments
                           (model, state_batch, action_batch, signal_batch, theta_batch) and
                           should output a `signal_hat`
    """
    ave_loss1 = ave_loss2 = ave_loss3 = 0.0
    stdev = len(memory) / 5
    for _ in range(epochs):
        states, actions, signals, thetas = memory.weighted_sample(batch_size, stdev)
        state_batch = torch.FloatTensor(states)
        action_batch = torch.FloatTensor(actions)
        signal_batch = torch.FloatTensor(signals)
        theta_batch = torch.FloatTensor(thetas)

        # 1. enforce that the interface is decodable

        states_actions = torch.FloatTensor([])
        state_clone = state_batch.clone()
        for _ in range(timesteps):
            signal_hat = model.interface_policy(
                torch.concat((state_clone, theta_batch), 1)
            )
            action_hat = model.human_policy(torch.concat((state_clone, signal_hat), 1))
            states_actions = torch.concat((states_actions, state_clone, action_hat), 1)
            state_clone += action_hat
        theta_hat = model.decoder(states_actions)
        dec_loss = F.mse_loss(theta_hat, theta_batch)
        ave_loss2 += dec_loss.item()

        # 2. behavior clone model of the human

        state_signals = torch.concat((state_batch, signal_batch), 1)
        action_hat = model.human_policy(state_signals)
        hum_loss = F.mse_loss(action_hat, action_batch)
        ave_loss1 += hum_loss.item()

        # 3. enforce priors (if they exist)

        prior_loss = torch.FloatTensor([0.0])

        if prior_training_func is not None:
            prior_loss += prior_training_func(
                model, state_batch, action_batch, signal_batch, theta_batch
            )
            ave_loss3 += prior_weight * prior_loss.item()

        loss = non_prior_weight * (dec_loss + hum_loss) + prior_weight * prior_loss

        human_optim.zero_grad()
        decoder_optim.zero_grad()
        loss.backward()
        human_optim.step()
        decoder_optim.step()

    ave_loss1 /= epochs
    ave_loss2 /= epochs
    ave_loss3 /= epochs

    return ave_loss1, ave_loss2, ave_loss3


def conv_prior(
    model: LIMIT,
    states: torch.Tensor,
    actions: torch.Tensor,
    signals: torch.Tensor,
    thetas: torch.Tensor,
):
    """
    enforces convexity on the magnitude of the signal via `MSE(R(s, Θ), -R(s, -Θ))`
    """
    state_theta = torch.concat((states, thetas), 1)
    neg_state_theta = torch.concat((states, -thetas), 1)
    return F.mse_loss(
        model.interface_policy(state_theta), -model.interface_policy(neg_state_theta)
    )


def prop_prior(
    model: LIMIT,
    states: torch.Tensor,
    actions: torch.Tensor,
    signals: torch.Tensor,
    thetas: torch.Tensor,
    eps=1e-1,
    gamma=1e-2,
):
    """
    Enforces proportionality as `f=e^{-ɣ||Θ-N(Θ,ε)||}(x - N(x,ε)) - x`
    """
    thetas_dist = Normal(thetas, eps)
    thetas_sampled = thetas_dist.rsample()
    state_thetas_sampled = torch.concat((states, thetas_sampled), 1)
    signals_sampled = model.interface_policy(state_thetas_sampled)
    return torch.sum(
        torch.exp(-gamma * F.mse_loss(thetas, thetas_sampled, reduction="none"))
        * F.mse_loss(signals, signals_sampled, reduction="none")
    )


def train_MLPHuman(
    model: MLPHuman, memory: ReplayMemory, optim: Adam, epochs=16, batch_size=256
):
    net_loss = 0.0
    stdev = len(memory) / 5
    for _ in range(epochs):
        states, _, signals, thetas = memory.weighted_sample(batch_size, stdev)
        state_batch = torch.FloatTensor(states)
        signal_batch = torch.FloatTensor(signals)
        theta_batch = torch.FloatTensor(thetas)
        action_hat = model(torch.concat((state_batch, signal_batch), 1))
        optimal_action = F.tanh(theta_batch - state_batch)
        loss = F.mse_loss(optimal_action, action_hat)
        optim.zero_grad()
        loss.backward()
        optim.step()
        net_loss += loss.item()
    return net_loss


def train_HighwayHuman(
    model: HighwayHuman,
    memory: ReplayMemory,
    optim: Adam,
    epochs=16,
    batch_size=256,
    n_thetas=4,
):
    net_loss = 0.0
    stdev = len(memory) / 3
    loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        states, _, signals, thetas = memory.weighted_sample(batch_size, stdev)
        state_batch = torch.FloatTensor(states)
        signal_batch = torch.FloatTensor(signals)
        theta_batch = torch.FloatTensor(thetas)
        state_signals = torch.concat((state_batch, signal_batch), dim=1)
        idx_probs = model.get_soft_theta(state_signals)
        actual_idxs = F.one_hot(theta_batch.long().squeeze(), num_classes=n_thetas)
        loss_ = loss(idx_probs, actual_idxs.float())
        optim.zero_grad()
        loss_.backward()
        optim.step()
        net_loss += loss_.item()
    return net_loss
