import torch
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from argparse import ArgumentParser, Namespace
from utils.models import (
    ReplayMemory,
    LIMIT,
    BayesOptWrapper,
    MLPHuman,
    HighwayHuman,
    DiscretePolicyA,
    DiscretePolicyB,
    DiscretePolicyC,
    DiscretePolicyD,
)
from utils.train import (
    train_bayes,
    train_LIMIT,
    train_MLPHuman,
    train_HighwayHuman,
    prop_prior,
    conv_prior,
)


def treasure(
    model: LIMIT | BayesOptWrapper,
    human: MLPHuman,
    memory: ReplayMemory,
    dof=2,
    timesteps=10,
    scale=20,
    bias=-10,
) -> float:
    """
    Performs an interaction in the `Treasure` environment.
    Here, `theta ~ U([scale + bias, scale - bias])`
    and `state_0 ~ U([scale + bias, scale - bias])`.

    Param      Use
    -----------------------------------------------
    `model`    : instance of LIMIT or BayesOptWrapper to
                 use for optimizing the interface
    `human`    : instance of simulated human agent
    `memory`   : ReplayMemory buffer, shared between
                 the interface optimizer and the human
    `dof`      : dimension of the (s, a, x, t) spaces
    `timesteps`: number of timesteps per interaction
    `scale`    : parameter for the (s, a, t) spaces
    `bias`     : parameter for the (s, a, t) spaces

    Returns
    -----------------------------------------------
    `cost`     : the final error between theta and state
    """
    if type(model) is LIMIT:
        theta = scale * torch.rand(dof) + bias
        state = scale * torch.rand(dof) + bias
        for _ in range(timesteps):
            state_theta = torch.concat((state, theta))
            signal = model.interface_policy(state_theta)
            state_signal = torch.concat((state, signal))
            action = human(state_signal)
            memory.push(
                state.detach().numpy(),
                action.detach().numpy(),
                signal.detach().numpy(),
                theta.detach().numpy(),
            )
            state += action
        return float(torch.linalg.norm(state - theta).item())
    elif type(model) is BayesOptWrapper:
        theta = scale * np.random.rand(dof) + bias
        state = scale * np.random.rand(dof) + bias
        for _ in range(timesteps):
            state_theta = np.concatenate((state, theta))
            signal = model(state_theta)
            state_signal = np.concatenate((state, signal))
            action = human(torch.FloatTensor(state_signal)).detach().numpy()
            memory.push(
                state,
                action,
                signal,
                theta,
            )
            state += action
        return float(np.linalg.norm(state - theta).item())
    else:
        raise NotImplementedError(f"Model {type(model)} is not implemented.")


def highway(
    model: LIMIT | BayesOptWrapper,
    human: HighwayHuman,
    memory: ReplayMemory,
    timesteps=4,
) -> float:
    """
    Performs an interaction in the `Highway` environment.
    Here, `theta ~ B([0, 4])`, `human_state ~ B([0, 1])`,
    and `robot_state = 1 - human_state`.

    Param      Use
    -----------------------------------------------
    `model`    : instance of LIMIT or BayesOptWrapper to
                 use for optimizing the interface
    `human`    : instance of simulated human agent
    `memory`   : ReplayMemory buffer, shared between
                 the interface optimizer and the human
    `timesteps`: number of timesteps per interaction

    Returns
    -----------------------------------------------
    `collision-rate`: the number of timesteps where
                      the human and the robot were
                      in collision, divided by the
                      number of total timesteps
    """
    error = 0.0
    human_state = torch.randint(0, 2, (1,)).float()
    robot_state = 1 - human_state
    theta = torch.randint(0, 4, (1,))
    policy = [
        DiscretePolicyA(),
        DiscretePolicyB(),
        DiscretePolicyC(),
        DiscretePolicyD(),
    ][theta]
    if type(model) is LIMIT:
        for _ in range(timesteps):
            state = torch.concat((human_state, robot_state))
            state_theta = torch.concat((state, theta))
            signal = model.interface_policy(state_theta)
            state_signal = torch.concat((state, signal))
            human_action = human(state_signal)
            robot_action = policy(human_state)
            memory.push(
                state.detach().numpy(),
                human_action.detach().numpy(),
                signal.detach().numpy(),
                theta.detach().numpy(),
            )
            state = torch.concat((human_action, robot_action))
            error += float(human_action == robot_action)
        return error / timesteps
    elif type(model) is BayesOptWrapper:
        for _ in range(timesteps):
            state = np.concatenate((human_state, robot_state))
            state_theta = np.concatenate((state, theta))
            signal = model(state_theta)
            state_signal = np.concatenate((state, signal))
            human_action = human(torch.FloatTensor(state_signal)).detach().numpy()
            robot_action = policy(human_state).detach().numpy()
            memory.push(
                state,
                human_action,
                signal,
                theta,
            )
            state = np.concatenate((human_action, robot_action))
            error += float(human_action[0] == robot_action[0])
        return error / timesteps
    else:
        raise NotImplementedError(f"Model {type(model)} is not implemented.")


def main(args: Namespace) -> None:
    """
    Main runner for all simulations.
    options:
      -h, --help            show this help message and exit
      --env {treasure,highway}
                            Environment to use.
      --model {bayes,limit,ours-c,ours-p,ours-pc}
                            Interface Optimizer to use.
      --dof DOF             DoF for the treasure environment.
                            Ignored for highway environment.
      --lr LR               Learning Rate for interface and human models
      --batch-size BATCH_SIZE
                            Batch size for interface and human models
      --epochs EPOCHS       Number of epochs to train interface and human
      --episodes EPISODES   Number of episodes to perform interaction
      --prior-weight PRIOR_WEIGHT
                            Weight of prior

    Note that `--prior-weight` applies `1 - prior-weight` to the
    LIMIT tuning algorithm: a prior weight of 1.0 indicates that
    the interface should *not* be tuned and only the prior
    should affect the signals of the interface.
    """
    memory = ReplayMemory(capacity=100_000)

    # determine environment

    if args.env == "treasure":
        state_dim = action_dim = signal_dim = theta_dim = args.dof
        timesteps = 10
        human = MLPHuman(
            state_dim, action_dim, signal_dim, state_dim + signal_dim, 1, F.relu
        )
        optim = Adam(human.parameters(), lr=args.lr)
        train_human = lambda: train_MLPHuman(
            human, memory, optim, epochs=args.epochs, batch_size=args.batch_size
        )
        interaction = lambda *args: treasure(*args, dof=state_dim, timesteps=timesteps)

    elif args.env == "highway":
        state_dim = 2
        action_dim = signal_dim = theta_dim = 1
        timesteps = 4
        human = HighwayHuman()
        optim = Adam(human.parameters(), lr=args.lr)
        train_human = lambda: train_HighwayHuman(
            human, memory, optim, epochs=args.epochs, batch_size=args.batch_size
        )
        interaction = lambda *args: highway(*args, timesteps=timesteps)

    else:
        raise NotImplementedError(f"Environment {args.env} not implemented.")

    # determine interface optimizer to use

    if args.model == "bayes":
        model = BayesOptWrapper(
            allow_duplicate_points=True,
            state_dim=state_dim,
            action_dim=action_dim,
            signal_dim=signal_dim,
            theta_dim=theta_dim,
        )
        train_model = lambda cost: train_bayes(-cost, model)
    else:
        model = LIMIT(
            state_dim=state_dim,
            action_dim=action_dim,
            signal_dim=signal_dim,
            theta_dim=theta_dim,
            timesteps=timesteps,
        )
        human_optim = Adam(model.human_policy_network.parameters(), lr=args.lr)
        decoder_optim = Adam(
            [
                {"params": model.decoder.parameters()},
                {"params": model.interface_policy_network.parameters()},
            ],
            lr=args.lr,
        )
        if args.model == "limit":
            prior_training_func = None
        elif args.model == "ours-c":
            prior_training_func = lambda *args: conv_prior(*args)
        elif args.model == "ours-p":
            prior_training_func = lambda *args: prop_prior(*args)
        elif args.model == "ours-pc":
            prior_training_func = lambda *args: prop_prior(*args) + conv_prior(*args)
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
        train_model = lambda _: train_LIMIT(
            model,
            memory,
            human_optim,
            decoder_optim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            prior_weight=args.prior_weight,
            non_prior_weight=1 - args.prior_weight,
            prior_training_func=prior_training_func,
            timesteps=timesteps,
        )

    errors = []
    with tqdm.trange(args.episodes) as t:
        for _ in t:
            error = interaction(model, human, memory)
            if len(memory) > args.batch_size:
                train_model(error)
                train_human()
            errors.append(error)
            t.set_description(f"Error: {error:2.2f}")

    N = max(50, int(args.episodes / 20))
    mu = np.convolve(errors, np.ones(N) / N, mode="valid")
    sigma = np.std(mu)
    plt.plot(mu)
    plt.fill_between(np.arange(len(mu)), mu - sigma, mu + sigma, alpha=0.2)
    plt.show()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["treasure", "highway"],
        default="treasure",
        help="Environment to use.",
    )
    parser.add_argument(
        "--model",
        choices=["bayes", "limit", "ours-c", "ours-p", "ours-pc"],
        default="limit",
        help="Interface Optimizer to use.",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=2,
        help="DoF for the treasure environment. Ignored for highway environment.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning Rate for interface and human models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for interface and human models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Number of epochs to train interface and human",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to perform interaction",
    )
    parser.add_argument(
        "--prior-weight", type=float, default=0.001, help="Weight of prior"
    )
    main(parser.parse_args())
