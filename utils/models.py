from bayes_opt import (
    BayesianOptimization,
    SequentialDomainReductionTransformer,
    UtilityFunction,
)
import os
import numpy as np
from scipy.stats import halfnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class BayesOptWrapper(object):
    def __init__(
        self,
        state_dim=2,
        signal_dim=2,
        action_dim=2,
        theta_dim=2,
        kappa=5e-2,
        xi=2e-2,
        min_win=0.0,
        allow_duplicate_points=False,
    ) -> None:
        current_matrix = np.empty((signal_dim, state_dim + theta_dim))
        self.signal_dim = signal_dim
        self.state_dim = state_dim
        self.theta_dim = theta_dim
        self.action_dim = action_dim
        self.xi = xi
        self.kappa = kappa
        # generate bounds dictionary
        bounds = (-1, 1)
        pbounds = {}
        iternum = 0
        for r in current_matrix:
            for _ in r:
                pbounds[str(iternum)] = bounds
                iternum += 1
        if min_win > 0.0:
            self.optimizer = BayesianOptimization(
                f=lambda: None,
                pbounds=pbounds,
                verbose=0,
                bounds_transformer=SequentialDomainReductionTransformer(
                    minimum_window=min_win
                ),
                allow_duplicate_points=allow_duplicate_points,
            )
        else:
            self.optimizer = BayesianOptimization(
                f=lambda: None,
                pbounds=pbounds,
                verbose=0,
                allow_duplicate_points=allow_duplicate_points,
            )
        self.utility = UtilityFunction(kind="ucb", kappa=self.kappa)
        self.current_matrix_params = self.optimizer.suggest(self.utility)

    def get_matrix(self, **kwargs) -> np.ndarray:
        return get_matrix_from_kwargs(
            state_dim=self.state_dim,
            theta_dim=self.theta_dim,
            signal_dim=self.signal_dim,
            **kwargs
        )

    def save_matrix(self, path: str):
        dir = path[0 : path.rindex("/")]
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass
        with open(path, "wb") as fh:
            pickle.dump(self.current_matrix_params, fh)
        return

    def load_matrix(self, path: str):
        with open(path, "rb") as fh:
            self.current_matrix_params = pickle.load(fh)
        return

    def __call__(self, state_theta: np.ndarray) -> np.ndarray:
        return np.tanh(self.get_matrix(**self.current_matrix_params)) @ state_theta


def get_matrix_from_kwargs(state_dim=1, theta_dim=1, signal_dim=1, **kwargs):
    """
    used for bayesian optimization to convert kwargs dict to a matrix
    """
    shape = (signal_dim, state_dim + theta_dim)
    mapping = np.array(list(kwargs.values()))
    mapping = mapping.reshape(shape)
    return mapping


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def save_model(m, path):
    dir = path[0 : path.rindex("/")]
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass
    torch.save(m.state_dict(), path)


def load_model(c, path, *args, **kwargs):
    """
    `c`: class
    `path`: path to load from
    `*args`: args to pass to class initialization
    `**kwargs`: kwargs to pass to class initialization
    """
    model = c(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_hidden_layers,
        activation_func=None,
        last_activation_func=True,
    ):
        """
        Simple QNetwork mapping arbitrary inputs to outputs. This network forms
        many of the deterministic networks herein.
        """
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layerf = nn.Linear(hidden_dim, output_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )

        if activation_func is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = activation_func

        if not last_activation_func:
            self.last_activation_func = nn.Identity()
        else:
            self.last_activation_func = torch.tanh

    def nm1_layer(self, x):
        x = self.activation_func(self.layer1(x))
        for layer in self.hidden_layers:
            x = self.activation_func(layer(x))
        return x

    def forward(self, x):
        x = self.nm1_layer(x)
        return self.last_activation_func(self.layerf(x))


class MLPHuman(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        signal_dim,
        hidden_dim,
        n_hidden_layers,
        activation_func=None,
    ):
        super(MLPHuman, self).__init__()
        self.policy = QNetwork(
            state_dim + signal_dim,
            action_dim,
            hidden_dim,
            n_hidden_layers,
            activation_func=activation_func,
            last_activation_func=False,
        )
        self.apply(weights_init_)

    def forward(self, state_signal):
        return torch.tanh(self.policy.forward(state_signal))


class HighwayHuman(nn.Module):
    """
    MLPHuman attempts to interpret theta_hat from signal, where theta hat is a
    representation of theta such that the human learns to take the appropriate
    predefined policy.
    """

    def __init__(self, state_dim=2, signal_dim=1):
        super(HighwayHuman, self).__init__()
        self.n_thetas = 4
        self.hidden_dim = 2
        self.state_dim = state_dim
        self.signal_dim = signal_dim
        self.linear1 = nn.Linear(self.state_dim + self.signal_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.n_thetas)
        self.policies = np.array(
            [
                DiscretePolicyB(),
                DiscretePolicyA(),
                DiscretePolicyD(),
                DiscretePolicyC(),
            ],
            dtype=object,
        )
        self.apply(weights_init_)

    def get_soft_theta(self, state_signal):
        x = self.linear1(state_signal)
        x = self.linear3(x)
        return x

    def forward(self, state_signal):
        x = self.get_soft_theta(state_signal)
        theta_hat = F.gumbel_softmax(x, tau=1, hard=True)
        policy_idxs = torch.argmax(theta_hat)
        policy = self.policies[policy_idxs.detach().numpy()]
        state = state_signal[0].unsqueeze(0)
        return policy(state)


class LIMIT(nn.Module):
    def __init__(
        self,
        theta_dim=1,
        state_dim=1,
        action_dim=1,
        signal_dim=1,
        hidden_dim1=8,
        hidden_dim2=16,
        n_hidden_layers1=3,
        n_hidden_layers2=3,
        timesteps=10,
        activation_func=None,
    ) -> None:
        """
        LIMIT: Consists of a decoder, interface policy, and a human policy
        """
        super(LIMIT, self).__init__()
        self.interface_policy_network = QNetwork(
            state_dim + theta_dim,
            signal_dim,
            hidden_dim1,
            n_hidden_layers1,
            activation_func=activation_func,
            last_activation_func=True,
        )
        self.human_policy_network = QNetwork(
            state_dim + signal_dim,
            action_dim,
            hidden_dim1,
            n_hidden_layers1,
            activation_func=activation_func,
            last_activation_func=True,
        )
        self.decoder = QNetwork(
            timesteps * (state_dim + action_dim),
            theta_dim,
            hidden_dim2,
            n_hidden_layers2,
            activation_func=None,
            last_activation_func=False,
        )

    def forward(self, state, theta):
        state_theta = torch.concat((state, theta), 1)
        signal = self.interface_policy(state_theta)
        state_signal = torch.concat((state, signal), 1)
        action = self.human_policy(state_signal)
        return action

    def interface_policy(self, state_theta):
        return self.interface_policy_network(state_theta)

    def human_policy(self, state_signal):
        return self.human_policy_network(state_signal)


class RobotPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers) -> None:
        super(RobotPolicy, self).__init__()
        self.policy = QNetwork(
            input_dim, output_dim, hidden_dim, n_hidden_layers, activation_func=F.relu
        )

    def forward(self, x):
        return F.tanh(self.policy(x))


class ReplayMemory(object):
    def __init__(self, capacity=1000):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.buffer = np.zeros(self.capacity, dtype=tuple)

    def push(self, *args):
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer[0 : self.size], batch_size)
        args = map(np.stack, zip(*batch))
        return args

    def weighted_sample(self, batch_size, stdev=10.0):
        weights = np.array(
            halfnorm.pdf(np.arange(0, self.capacity), loc=0, scale=stdev)
        )
        weights = weights.take(np.arange(self.position - self.capacity, self.position))[
            ::-1
        ][0 : self.size]
        weights /= np.sum(weights)
        batch = np.random.choice(self.buffer[0 : self.size], batch_size, p=weights)
        args = map(np.stack, zip(*batch))
        return args

    # length function, returns number of datapoints in the buffer
    def __len__(self):
        return self.size


class DiscretePolicy(object):
    """
    Always returns the state. Here, `state` refers to the **opponent's** state
    """

    def __init__(self, state=None) -> None:
        if state is None:
            self.state = np.random.binomial(1, 0.5)

    def reset(self, state=None):
        self.__init__(state=state)

    def __call__(self, state):
        """
        Always returns the state
        Here, `state` refers to the **opponent's** state
        """
        return state


class DiscretePolicyA(DiscretePolicy):
    """
    Always returns 0.0.
    """

    def __call__(self, _):
        """
        Always returns 0.0.
        """
        return torch.FloatTensor([0.0])


class DiscretePolicyB(DiscretePolicy):
    """
    Always returns 1.0.
    """

    def __call__(self, _):
        """
        Always returns 1.0.
        """
        return torch.FloatTensor([1.0])


class DiscretePolicyC(DiscretePolicy):
    """
    Return the opposite state
    Here, `state` refers to the **opponent's** state
    """

    def __call__(self, state):
        """
        Returns the opposite of the state.
        Here, `state` refers to the **opponent's** state
        """
        return 1 - state


class DiscretePolicyD(DiscretePolicy):
    """
    Always returns the state
    """

    pass
