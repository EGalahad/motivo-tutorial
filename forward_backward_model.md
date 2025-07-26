# Forward Backward Model

In this note, I will introduce an unsupervised RL algorithm called forward backward model from the view of multi-task policy iteration. The insight is that forward backward model is essentially dynamics-only policy evaluation plus multi-task policy improvement. By factorizing reward and dynamics and only representing the dynamics of the MDP, FB model enables policy improvement step to be conducted on a family of reward functions simultaneously.

Apart from the elegant math formulation, FB model can also be intuitively understood as a latent space goal conditioned reinforcement learning framework, where the latent space is automatically constructed with dynamics prediction as the self-supervision signal.

Note: This note only covers the forward backward model formulation from a new perspective, aiming to provide intuition and understanding of the algorithm. It does not cover the optimization process (how the learning converges), is agnostic optimization algorithm involved (whether is is PPO or SAC).

Some relevant papers: Meta Motivo, HILP, 

## Occupancy Measure and Q Function

Let's start with the definition of occupancy measure and Q function.

### Q Function
The definition of the Q function:

$$Q^{\pi}(s,a) = E_{\tau \sim \pi}[\gamma^t \cdot r(s_t)]$$

where $\tau$ follows the trajectory of the MDP and following the policy.

The bellman equation for Q function. 

$$Q^{\pi}(s,a) = r(s,a) + \gamma \cdot E_{s', a'}[Q^{\pi}(s',a')]$$

The expectation is over the next state by $s' \sim P(s'|s,a)$ and the next action by $a' \sim \pi(a'|s')$.

Note: we define a Q function with **an MDP with reward and a policy**. It captures the discounted future reward starting from $s$ and $a$ and following the policy.


### Occupancy Measure
The definition of occupancy measure:

$$M^{\pi}(s^+ | s, a) = E_{\tau \sim \pi}[\gamma^t \cdot P(s_t = s^+)]$$

The bellman equation for occupancy measure:

$$M^{\pi}(s^+ | s, a) = p(s^+ | s, a) + \gamma \cdot E_{s', a'}[M^{\pi}(s^+ | s', a')]$$

here $M^{\pi}(s^+ | s, a)$ is the pdf version of the occupancy measure.

Note: we define an occupancy measure with **a reward-free MDP and a policy**. It captures the discounted future distribution of the states starting from $s$ and $a$ and following the policy.

#### Relationship between Q function and occupancy measure

Though the occupancy measure is defined with a reward-free MDP, given arbitrary reward function $r(s^+)$, we can use the occupancy measure to evaluate the Q function to that reward function $r(s^+)$: since the occupancy measure is the expected discounted future state distribution, we can use it to evaluate the Q function by integrating over all possible future states and weighting them by the reward function.
$$Q^{\pi}(s,a) = \int_{s^+} M^{\pi}(s^+ | s, a) \cdot r(s^+)$$

> **An interesting derivation of the Bellman equation for occupancy measure**
> 
> If we write the bellman equation for Q function and replace the Q function with the occupancy measure, we get:
> $$\int_{s^+} M^{\pi}(s^+ | s, a) \cdot r(s^+) = \int_{s^+} p(s^+ | s, a) r(s^+) + \gamma \cdot E_{s', a'}\left[ \int_{s^+} M^{\pi}(s^+ | s', a') \cdot r(s^+) \right]$$
>
> Let's abuse math a bit. Since the above equation should hold true for all possible reward functions $r(s^+)$, we can remove the integral among the reward function and the occupancy measure. Or more rigorously, we can use a delta function as the reward function. Both give us the same equation:
>
> $$M^{\pi}(s^+ | s, a) = p(s^+ | s, a) + \gamma \cdot E_{s', a'}\left[ M^{\pi}(s^+ | s', a') \right]$$

## From Q-learning to Multi-task Policy Iteration

### Q-learning

Recall the Q-learning algorithm, which does Q function bellman updates in the following form:

$$Q(s,a) \to r(s) + \gamma \max_{a'} \mathbb{E}_{s' \sim P(s'|s,a)}[Q(s',a')]$$

In the discrete case, the Q function itself is a representation of the optimal policy, since we can take the argmax of the Q function to get the optimal action.

$$\pi(s) = \argmax_{a} Q(s,a)$$

### DDPG

In continuous action spaces, we cannot take that argmax, we need to have a policy to propose the action. If we think of argmax as a rule based selection, using a policy to propose the action is a neural approximation of the same operation.

So the algorithm's splits to two steps, one evaluating the current policy and one optimizing the policy.

$$Q(s,a) \to r(s) + \gamma \mathbb{E}_{a' \sim \pi(a'|s')} \mathbb{E}_{s' \sim P(s'|s,a)}[Q(s',a')]$$

$$\mathcal{L}(\pi) = - \mathbb{E}_{s}[Q(s,a)]$$

We abuse math notation and use the following notation to denote the policy improvement step:

$$\pi(s) \to \argmax_{a} Q(s,a)$$

### DDPG with Occupancy Measure

Now we can rewrite the above DDPG with occupancy measure:

$$M^\pi(s^+ | s, a) \to p(s^+ | s, a) + \gamma \mathbb{E}_{s', a'}\left[ M^\pi(s^+ | s', a') \right]$$

$$\pi(s) \to \argmax_{a} \int_{s^+} M^\pi(s^+ | s, a) \cdot r(s^+)$$

The first equation is the bellman equation for occupancy measure, it captures the policy related transition dynamics of a reward-free MDP. The second equation takes reward into account and optimizes the policy by maximizing the expected discounted future state visitation weighted by the reward it will receive at those future states.

What does this new interpretation from occupancy measure perspective give us? 
The first equation provides a way to summarize the transition dynamics of the MDP without considering rewards. This factorization allows us to learn the transition dynamics of the MDP in a reward-agnostic way, but we can still evaluate the Q function by integrating over all possible future states and weighting them by the reward function.
Only the second equation is related to the reward, we can plug in multiple reward functions $\{r_i\}$ to the second equation and optimize for the optimal policies for each reward function simultaneously.

### Occupancy Measure for Multi-task Policy Iteration (Abstract Formulation)

Now we can simply extend the above idea to multi-task policy iteration. Assume we are given a family of reward functions $\{r(s^+)\}$, we can learn a family of policies $\{\pi_r\}$ that each is optimal for a reward function $r$ by optimizing the following objective:
$$M^{\pi_r}(s^+ | s, a) \to p(s^+ | s, a) + \gamma \mathbb{E}_{s', a'}\left[ M^{\pi_r}(s^+ | s', a') \right] \text{ for each } r$$

$$\pi_r(s) \to \argmax_{a} \int_{s^+} M^{\pi_r}(s^+ | s, a) \cdot r(s^+) \text{ for each } r$$

## Forward Backward Model

<!-- Having laid out the foundation, we now introduce forward backward model. The goal is to provide a structured way to evaluate and optimize policies in a multi-task setting. Given a reward-free MDP, we want to unsupervisedly learn a family of reward functions and a family of policies that is optimal for these reward functions. To make a concrete algorithm, we will use a latent space $z$ to represent/index the reward function and the corresponding optimal policies. -->

### Definition of Forward Backward Model

Forward backward model factors the occupancy measure as:

$$M^{\pi}(s^+ | s, a) = F^\pi(s, a)^T B(s^+)$$

here $F^\pi(s, a)$ is the forward model and $B(s^+)$ is the backward model. Both project the state $s$ (and action $a$) to a latent vector $z$.

Note here $F^\pi(s, a)$ is associated with the policy and $B(s^+)$ is not associated with the policy.

Intuitively, we can think of $B$ as a encoder from state space into a latent space that captures some features of the state $s^+$. $F^\pi(s, a)$ measures the sum of discounted future latents of the policy under such an encoding scheme as dictated by $B$. We will explain more intuition of $F^\pi$ and $B$ later.

### Zero-shot Policy Evaluation

Now our goal is to find an approach that can do multi-task policy iteration. The first step is given a learned FB model for a policy, how can we evaluate the policy when provided with a reward function?

Given an arbitrary policy $\pi$, and an arbitrary reward function $r(s)$, we can evaluate the policy by:

$$Q^{\pi}(s,a) = \int_{s^+} M^{\pi}(s^+ | s, a) \cdot r(s^+) = F^\pi(s, a)^T \int_{s^+} B(s^+) \cdot r(s^+)$$

### Optimal Policy Analysis

Since our focus is multi-task, so we must learn a family of policies $\{\pi\}$. the desired property would be
1. The family of policies is expressive enough to be good for a large class of reward functions.
2. Given a reward function, the optimal policy for the reward function can be easily retrieved from the family of policies.

Now assume 1 is satisfied, how can we find the optimal policy for a given reward function? Lets further rewrite the Q function as:

$$Q^{\pi}(s,a) = F^\pi(s, a)^T z$$

where $z = \int_{s^+} B(s^+) \cdot r(s^+)$.

Naively $F^\pi(s, a)$ for the best policy is the one that aligns in direction with $z$ in the latent space.

...And its length be as long as possible?

#### Normalization of FB Model

We will get to a bit of math details here to answer the question.

Recall the definition of the occupancy measure, the discounted future state distribution, so it must integrate to 1.

$$\int_{s^+} M^{\pi}(s^+ | s, a) = 1$$

Factorized by the FB model, we get:

$$\int_{s^+} M^{\pi}(s^+ | s, a) =  F^\pi(s, a)^T \int_{s^+} B(s^+) = 1$$

So let's answer the question: if we increase the length of $F^\pi(s, a)$ by a constant, wishing we find a better policy, it is not the case because we must decrease the length of $B(s^+)$ by the same constant, so it still corresponds to the same policy.

There can be infinite solutions for $F^\pi(s, a)$ and $B(s^+)$ that satisfy the above equation just by scaling up/down $F$ and $B$ by the same constant. So we make it a rule that the length of $F^\pi(s, a)$ must be 1.


rule: $||F^\pi(s, a)^T z|| = 1$

So under this rule, we know the optimal policy for $z$ is such that $F^\pi(s, a)$ is a unit vector that points to the direction of $z$.
<!-- 
### Multi-Task Policy Iteration

Look at the formula of $z$ again:

$$z = \int_{s^+} B(s^+) \cdot r(s^+)$$

If we view it as a weighted sum of the latents it will make sense a lot why $F^\pi(s, a)^T z$ is the Q function. For each state $s^+$, we weight it by the reward function. so if a state is more important in the reward function, it will contribute more to the direction of $z$. In this view, maximizing $F^\pi(s, a)$ towards this direction is likely to maximize the probability of encoutering high reward region in the future.


So we can use the following loss function to optimize the policy:

$$\mathcal{L}(\pi) = - \mathbb{E}_{s}[F^\pi(s, a)^T z]$$


Let's frame our goal again:

Given a reward-free MDP, we want to learn a family of policies, such that given any reward function at inference time, we can find the optimal policy for the reward function. We parameterize the family of policies by $z$, and we want the policy $\pi_z$ to be the optimal policy for the reward function $r_z$. -->


### Multi-Task Policy Iteration
Having laid out the foundation, we now introduce forward backward model for multi-task policy iteration. The goal is to provide a structured way to evaluate and optimize policies in a multi-task setting. Given a reward-free MDP, we want to unsupervisedly learn a family of reward functions and a family of policies that is optimal for these reward functions. To make a concrete algorithm, we will use a latent space $z$ to represent the reward function and the corresponding optimal policies.

We need a mapping from reward to the latent space. Assume we have already learned the FB model, we can use the backward encoder to map the reward function to the latent space:
$$z = \int_{s^+} B(s^+) \cdot r(s^+) $$

Now the Q function for the policy $\pi_z$ is:
$$Q^{\pi_z}(s,a) = F^{\pi_z}(s, a)^T \int_{s^+} B(s^+) \cdot r(s^+) = F^{\pi_z}(s, a)^T z$$

The bellman update and policy improvement step is:
$$F^{\pi_z}(s, a) B(s^+) \to p(s^+ | s, a) + \gamma \mathbb{E}_{s', a'}\left[ F^{\pi_z}(s', a')^T B(s^+) \right]$$

$$\pi_z(s) \to \argmax_{a}  F^{\pi_z}(s, a)^T z$$

In pratice, we use a neural network conditioned on the reward latent $z$, $F(s, a, z)$ to represent the forward model $F^{\pi_z}(s, a)$. The policy is also represented by a reward latent conditioned neural network $\pi(s, z)$.

The first equation is just bellman update for occupancy measure $M$. How to understand the second equation? Why optimizing $F$ dot-product with $z$?


Let’s look at the thing in the argmax operator, which is the Q function.

$$Q^{\pi_z}(s,a) = F^{\pi_z}(s, a)^T \int_{s^+} B(s^+) \cdot r(s^+) = F^{\pi_z}(s, a)^T z$$

Consider the simple sketch of the state space below.
![alt](reward_in_state_space.png)

z is actually the sum of latent B(s) for states $s^+$ in the high reward region. It means some kind of goal in latent space.
Since occupancy measures summarize the future probability of visiting state $s^+$ from state $s$ and action $a$, F can be understood as discounted sum of future latents. So optimizing $pi$ to increase that dot product is to make future latents be closer to the high reward region.

## Summary

Intuition:
1. $B(s^+)$: maps explicit states into latent goal space
2. $F(s, a)$: predicts forwards dynamics from current state in the latent space
3. $z$: the projection of a reward function into the latent space
4. $\pi(s, a, z)$: the best policy for reward whose projection is $z$

Thus FB model can be understood as a latent space goal conditioned reinforcement learning framework, where the latent space is automatically constructed with dynamics prediction/occupancy measure approximation as the self-supervision signal.



