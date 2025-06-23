# Forward Backward Model

In this note, I will introduce forward backward model for unsupervised RL from the view of value iteration and policy iteration. The insight is that forward backward model is a multi-task and dynamics-only version of value iteration and policy iteration, by factorizing the reward and dynamics and only representing the dynamics of the MDP. 

The representation of FB model makes it possible to do zero-shot policy evalutation, and enables to train a family of policies to optimize rewards that can be projected onto a latent space.

Some relevant papers: Meta Motivo, HILP

## Occupancy Measure and Q Function

Let's start with the definition of occupancy measure and Q function.

### Q Function
The definition of the Q function:

\[Q^{\pi}(s,a) = E_{\tau \sim \pi}[\gamma^t \cdot r(s_t)]\]

where \(\tau\) is the follows the trajectory of the MDP and following the policy.

The bellman equation for Q function. 

\[Q^{\pi}(s,a) = r(s,a) + \gamma \cdot E_{s', a'}[Q^{\pi}(s',a')]\]

The expectation is over the next state by \(s' \sim P(s'|s,a)\) and the next action by \(a' \sim \pi(a'|s')\).

Note: we define a Q function with **an MDP with reward and a policy**. It captures the discounted future reward starting from \(s\) and \(a\) and following the policy.


### Occupancy Measure
The definition of occupancy measure:

\[M^{\pi}(s^+ | s, a) = E_{\tau \sim \pi}[\gamma^t \cdot P(s_t = s^+)]\]

The bellman equation for occupancy measure:

\[M^{\pi}(s^+ | s, a) = p(s^+ | s, a) + \gamma \cdot E_{s', a'}[M^{\pi}(s^+ | s', a')]\]

here \(M^{\pi}(s^+ | s, a)\) is the pdf version of the occupancy measure.

Note: we define an occupancy measure with **a reward-free MDP and a policy**. It captures the discounted future distribution of the states starting from \(s\) and \(a\) and following the policy.


> **Another view of the Bellman equation for occupancy measure**
>
> We can represent the Q function as a function of the occupancy measure, since one is the expected discounted future reward and the other is the discounted future states, we can just multiply the occupancy measure with the reward and take the expectation over all possible future states:
> \[Q^{\pi}(s,a) = E_{s^+} [M^{\pi}(s^+ | s, a) \cdot r(s,a)]\]
> If we write the bellman equation for Q function and replace the Q function with the occupancy measure, we get:
> \[\int_{s^+} M^{\pi}(s^+ | s, a) \cdot r(s^+) = \int_{s^+} p(s^+ | s, a) r(s^+) + \gamma \cdot E_{s', a'}\left[ \int_{s^+} M^{\pi}(s^+ | s', a') \cdot r(s^+) \right]\]
>
> Let's abuse math a bit. Since the above equation should hold true for all possible reward functions \(r(s^+)\), we can remove the integral among the reward function and the occupancy measure. Or more rigorously, we can use a delta function as the reward function. Both give us the same equation:
>
> \[M^{\pi}(s^+ | s, a) = p(s^+ | s, a) + \gamma \cdot E_{s', a'}\left[ M^{\pi}(s^+ | s', a') \right]\]

## Value Iteration and Policy Iteration

Let's put occupancy measure aside and think about how can we get an optimal policy given a MDP with a reward function \(r(s)\).

Assume we already have an optimal policy \(\pi^*(s)\) then the Q function associated with the optimal policy \(Q^*(s,a)\) must satisfy the Bellman optimality equation:

\[Q^*(s,a) = r(s) + \gamma \max_{a'} \mathbb{E}_{s' \sim P(s'|s,a)}[Q^*(s',a')]\]


### Discrete case

In the discrete case, the Q function itself is a representation of the optimal policy, since we can take the argmax of the Q function to get the optimal action.

\[\pi^*(s) = \arg\max_{a} Q^*(s,a)\]

The Bellman optimality operator \(\mathcal{T}\) is defined as:

\[(\mathcal{T}Q)(s,a) = r(s) + \gamma \max_{a'} \mathbb{E}_{s' \sim P(s'|s,a)}[Q(s',a')]\]

We can mathematically prove that if we apply the Bellman optimality operator iteratively, the Q function will converge to the optimal Q function.

This is value iteration or DQN.

### Continuous case

In continuous action spaces, to take that argmax, we need to have a policy to propose the action.

If we think of argmax as a rule based selection, using a policy to propose the action is a approximate way to the same thing.

So the policy is optimized according to the following loss function:

\[\mathcal{L}(\phi) = - \mathbb{E}_{s}[Q(s,a)]\]

It can be proved that the policy and the Q function will converge to the optimal policy and the optimal Q function. I don't know how to prove it haha.

This is policy iteration or DDPG.

## Forward Backward Model

Having layed out the foundation, we now introduce the forward backward model.

### Definition of Forward Backward Model

The forward backward model factors the occupancy measure as:

\[M^{\pi}(s^+ | s, a) = F^\pi(s, a)^T B(s^+)\]

here \(F^\pi(s, a)\) is the forward model and \(B(s^+)\) is the backward model and both project the state (and action) to a vector in the latent space.

Note here \(F^\pi(s, a)\) is associated with the policy and \(B(s^+)\) is not associated with the policy. 

Intuitively, we can think of \(B\) as a unified encoder from state space into a latent space and \(F^\pi(s, a)\) measures the sum of discounted future latents of the policy under such an encoding scheme as dictated by \(B\). We will explain more intuition of \(F^\pi\) and \(B\) later.

### Zero-shot Policy Evaluation

Now our goal is to find an approach that can do multi-task policy iteration. The first step is given a learning FB model for a policy, how can we evaluate the policy when provided with a reward function?

Given an arbitrary policy \(\pi\), and an arbitrary reward function \(r(s)\), we can evaluate the policy by:

\[Q^{\pi}(s,a) = \int_{s^+} M^{\pi}(s^+ | s, a) \cdot r(s^+) = F^\pi(s, a)^T \int_{s^+} B(s^+) \cdot r(s^+)\]

### Optimal Policy Analysis

Since our focus is multi-task, so we must learn a family of policies \(\{\pi\}\). the desired property would be
1. The family of policies is expressive enough to be good for a large class of reward functions.
2. Given a reward function, the optimal policy for the reward function can be easily retrieved from the family of policies.

Now assume 1 is satisfied, how can we find the optimal policy for a given reward function? Lets furtur rewrite the Q function as:

\[Q^{\pi}(s,a) = F^\pi(s, a)^T z\]

where \(z = \int_{s^+} B(s^+) \cdot r(s^+)\).

Naively \(F^\pi(s, a)\) for the best policy is the one that aligns in direction with \(z\) in the latent space.

...And its length be as long as possible?

#### Normalization of FB Model

We will get to a bit of math details here to answer the question.

Recall the definition of the occupancy measure, the discounted future state distribution, so it must integrate to 1.

\[\int_{s^+} M^{\pi}(s^+ | s, a) = 1\]

Factorized by the FB model, we get:

\[\int_{s^+} M^{\pi}(s^+ | s, a) =  F^\pi(s, a)^T \int_{s^+} B(s^+) = 1\]

So let's answer the question: if we increase the length of \(F^\pi(s, a)\) by a constant, wishing we find a better policy, it is not the case because we must decrease the length of \(B(s^+)\) by the same constant, so it still corresponds to the same policy.

There can be infinite solutions for \(F^\pi(s, a)\) and \(B(s^+)\) that satisfy the above equation just by scaling up/down \(F\) and \(B\) by the same constant. So we make it a rule that the length of \(F^\pi(s, a)\) is 1.


rule: \(||F^\pi(s, a)^T z|| = 1\)

So under this rule, we know the best policy is such that \(F^\pi(s, a)^T z\) is a unit vector that points to the direction of \(z\).

### Multi-Task Policy Iteration

Now we know, in this latent space, only latent direction matters. \(F^\pi(s, a)\) describes the direction of the policy is heading to in the latent space.

Look at the formula of \(z\) again:

\[z = \int_{s^+} B(s^+) \cdot r(s^+)\]

If we view it as a weighted sum of the latents it will make sense a lot why \(F^\pi(s, a)^T z\) is the Q function. For each state \(s^+\), we weight it by the reward function. so if a state is more important in the reward function, it will contribute more to the direction of \(z\). In this view, maximizing \(F^\pi(s, a)\) towards this direction is likely to maximize the probability of encoutering high reward region in the future.


So we can use the following loss function to optimize the policy:

\[\mathcal{L}(\pi) = - \mathbb{E}_{s}[F^\pi(s, a)^T z]\]


Let's frame our goal again:

Given a reward-free MDP, we want to learn a family of policies, such that given any reward function at inference time, we can find the optimal policy for the reward function. We parameterize the family of policies by \(z\), and we want the policy \(\pi_z\) to be the optimal policy for the reward function \(r_z\).






