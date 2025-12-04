# Processor

The **Processor** node serves as Jacinta’s internal stochastic decision mechanism. It maintains and adapts a multivariate Gaussian distribution $\mathcal{N}(\mu, \Sigma)$, which functions as a compact yet expressive policy over continuous outputs. By repeatedly sampling from this distribution and updating it using reward feedback, the Processor gradually shifts its attention toward more promising regions of the output space, allowing it to learn even in the absence of gradients or structural information about the environment.

Two components define the Processor’s behavior:

- The mean vector $\mu$ represents the expected output and acts as the center of gravity of the distribution.
- The covariance matrix $\Sigma$ determines how broadly the system explores around that mean, balancing exploration and exploitation through its variances.

With these foundations in place, the Processor becomes an adaptable search engine for continuous domains. It can operate independently or serve as a building block within more elaborate architectures such as the **Cell** node.

## TODOs

The current implementation is intentionally minimal, and several planned improvements aim to expand its reliability and capabilities:

- Complete documentation once the API stabilizes.
- Comprehensive unit testing for deterministic and stochastic behavior.
- Configurable learning styles that range from conservative to aggressive.
- Support for alternative reward normalization or shaping techniques.
- Improved argument validation and clearer error responses.
- Enhanced state encapsulation to prevent unintended external interference.
- Extended monitoring tools for inspecting internal learning dynamics.
- Formal empirical evaluations to characterize convergence properties.

## How to Use

A Processor can be instantiated directly and used as a standalone adaptive sampler. The basic usage pattern involves generating outputs and updating the distribution based on their rewards:

```python
from jacinta.nodes import Processor


processor = Processor(size=2)

# Generate an output sample from the current distribution
y = processor.process_forward()

# Update the distribution using a reward signal
processor.process_backward(y, r=1.0)
```

In this loop, the Processor behaves like a lightweight evolutionary search unit. It gradually reshapes its distribution solely on the basis of rewards, which enables it to adapt without the need for gradients or explicit knowledge of the environment’s dynamics.

## Behavior

The Processor’s behavior unfolds through the interaction of two tightly connected processes: stochastic sampling, which probes the output space, and reward-guided adaptation, which incorporates experience into the distribution.

### Sampling

`process_forward` samples an output from the current Gaussian. The shape of this distribution determines how the system explores. High variances result in samples that cover a wide region, encouraging discovery of unexpected or promising outputs. As learning progresses and the distribution finds profitable areas, the covariance tends to narrow, making the sampling process more focused and decisive. This mechanism allows exploration and exploitation to emerge naturally from the evolution of $\Sigma$, without the need for manually tuning exploration schedules or external heuristics.

### Learning from rewards

`process_backward` adjusts the distribution in response to the reward obtained for a sampled output. Rewards influence both the mean and the covariance, allowing the system not only to shift its expectations but also to reshape the space in which it explores.

When a sample receives a positive reward, the mean moves toward it, reflecting the intuition that outputs similar to that sample are desirable. At the same time, the covariance adapts depending on how surprising the sample was. If an output far from the mean performs well, the system benefits from keeping exploration wider in that direction; if a sample close to the mean receives a high reward, the opposite occurs and the distribution becomes more focused. These updates allow the Processor to continually refine its understanding of the landscape while keeping enough flexibility to escape local optima.

To ensure that learning remains stable even in noisy or shifting environments, rewards are centered and normalized using exponential moving averages. This prevents extreme values or sudden changes in scale from destabilizing the updates.

### Additional capabilities

The Processor includes practical utilities that make it suitable for larger systems. It supports full serialization to and from JSON, enabling persistence and reproducibility. Furthermore, it allows dimensions to be added or removed dynamically, which makes it possible to adapt the complexity of the output space as the surrounding architecture evolves.

## Mathematics

Mathematically, the Processor models its policy as a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$. Sampling relies on the reparameterization technique

$$
y = \mu + Lz,
$$

where $z \sim \mathcal{N}(0, I)$ and $L$ is the Cholesky factor of $\Sigma$. This formulation ensures that each sample respects the covariance structure while keeping the sampling process numerically efficient and differentiable in principle, even though gradients are not explicitly used here.

### Mean update

To update the mean, the Processor first considers the deviation between the sampled output and the current expectation,

$$
\mathrm{diff} = y - \mu.
$$

This vector indicates the direction in which the sample lies relative to the mean, but using it directly would make the update overly sensitive to scale and orientation. Instead, the Processor projects the deviation through the inverse covariance,

$$
v_\mu = \Sigma^{-1} \, \mathrm{diff},
$$

which effectively measures how unexpected the sample is when accounting for the current uncertainty. This transformation results in a direction that naturally reflects the geometry of the distribution. After normalizing this direction to control step size, the mean is updated as

$$
\mu \leftarrow \mu + \alpha_\mu \, r_{\mathrm{norm}} \, v_\mu.
$$

This formulation allows the mean to shift decisively toward rewarding regions while maintaining numerical stability.

### Covariance update

The covariance update is designed to adapt the uncertainty of the distribution based on the structure revealed by each sample. The Processor compares the outer product of the deviation vector,

$$
\mathrm{diff} \, \mathrm{diff}^T,
$$

with the current covariance $\Sigma$. This comparison produces a matrix

$$
v_\Sigma = \mathrm{diff} \, \mathrm{diff}^T - \Sigma,
$$

which indicates whether the observed variation along each direction is larger or smaller than what the distribution currently expects. If certain directions exhibit more variability than anticipated, the covariance increases in those directions. If variability is lower, the covariance decreases. The update

$$
\Sigma \leftarrow \Sigma + \alpha_\Sigma \, r_{\mathrm{norm}} \, v_\Sigma
$$

allows the distribution to reshape itself in a way that reflects the structure of the reward landscape. After the update, symmetry is enforced and minimal variance constraints are applied to ensure that the covariance remains numerically stable and positive definite.

### Reward normalization

Reward normalization is essential for stable learning. Using exponential moving averages, the Processor estimates both a baseline and a scale term. Centering the reward removes long-term drift, while dividing by the scale term prevents updates from exploding when absolute rewards become large:

$$
r_{\mathrm{centered}} = r - \bar{r}, \qquad
r_{\mathrm{norm}} = \frac{r_{\mathrm{centered}}}{s + \varepsilon}.
$$

This yields a normalized reward signal with a roughly consistent magnitude, which allows the learning rates to remain meaningful across different tasks and reward regimes.
