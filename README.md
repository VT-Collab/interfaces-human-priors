# Accelerating Interface Adaptation with User-Friendly Priors

This repository houses an open-source implementation of our paper "[Accelerating
Interface Adaptation with User-Friendly Priors](https://collab.me.vt.edu/pdfs/ben_iros2024.pdf)".

This repository implements two environments (**Treasure** and **Highway**) and four
interface optimizers (**Bayes**, **LIMIT**, *Convexity*, *Proportionality*).
The *Convexity* and *Proportionality* optimizers can be used with or without
the tuning algorithm **LIMIT**, see **Arguments** for details.

All dependencies required to run the code in this repository are tabulated in
`requirements.txt`; call `pip install -r requirements.txt` to install necessary
dependencies.

## Arguments

See below:
```
usage: main.py [-h] [--env {treasure,highway}] [--model {bayes,limit,ours-c,ours-p,ours-pc}] 
               [--dof DOF] [--lr LR] [--batch-size BATCH_SIZE] [--epochs EPOCHS] 
               [--episodes EPISODES] [--prior-weight PRIOR_WEIGHT]

options:
  -h, --help            show this help message and exit
  --env {treasure,highway}
                        Environment to use.
  --model {bayes,limit,ours-c,ours-p,ours-pc}
                        Interface Optimizer to use.
  --dof DOF             DoF for the treasure environment. Ignored for highway environment.
  --lr LR               Learning Rate for interface and human models
  --batch-size BATCH_SIZE
                        Batch size for interface and human models
  --epochs EPOCHS       Number of epochs to train interface and human
  --episodes EPISODES   Number of episodes to perform interaction
  --prior-weight PRIOR_WEIGHT
                        Weight of prior
```

## Models

The four interface optimizers can be combined in eight different ways:

- **Bayes**: follows the Bayesian Optimization process by treating the
  environment and the user as Gaussian Processes [\[1\]](https://www.biorxiv.org/content/10.1101/095190v3.full.pdf)
- **LIMIT**: uses an information-theoretic model to train the interface in a
  *task-agnostic* way [\[2\]](https://arxiv.org/pdf/2304.08539.pdf)
- *Convexity*: attempts to form a convex signal manifold for all $\theta \in \Theta$
- *Proportionality*: attempts to form *proportional* signals about the space $\Theta$

*Convexity* and *Proportionality* are combined to form the model "Ours-PC".
Note that the weight of the prior in training can be adjusted using the
"--prior-weight" flag, setting this value to $1.0$ causes the "non-prior"
weight to reduce to $0.0$. In this way, the performance of *just* the prior
can be observed relative to the models "Ours-C", "Ours-P", and "Ours-PC".


## Treasure

In the *Treasure* environment, a simulated agent is attempting to navigate to a
hidden goal position $\theta$. An interface attempts to signal the position
of this goal using $$x \sim \pi_R\left(\circ \mid s, \theta\right)$$ where $s$
is the simulated human's position in the environment. The human then takes
actions according to this signal: $$a \sim \pi_H\left(\circ \mid s, x\right)$$

The environment has dynamics: $$s^{t+1} = s^t + a^t$$

To launch this environment call `python main.py --env treasure --dof <dim>`
where *dim* specifies the dimension of the environment.

## Highway

Here, a simulated human is attempting to pass an autonomous vehicle. The
autonomous vehicle's policy is parameterized by hidden information $\theta$:
they are a mix of aggressive and defensive policies based on this hidden
information. Just as in **Treasure**, the interface attempts to convey the 
hidden information using signals $x \sim \pi_R\left(\circ \mid s, \theta\right)$.
Here, $s$ refers to the lane of the autonomous vehicle and the driver's vehicle.

The environment has dynamics $s^{t+1} = (a^t, u^t)$ where $u^t$ is the
autonomous vehicle's action at timestep $t$.

To launch this environment call `python main.py --env highway`.

### References

[1]: Schulz, Eric, Maarten Speekenbrink, and Andreas Krause. "A tutorial on Gaussian process regression: Modelling, exploring, and exploiting functions." Journal of Mathematical Psychology 85 (2018): 1-16.

[2]: Christie, Benjamin A., and Dylan P. Losey. "LIMIT: Learning interfaces to maximize information transfer." arXiv preprint arXiv:2304.08539 (2023).
