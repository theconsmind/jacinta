# Processor Module

## 🚧 Roadmap / Next Steps

The Processor module is functional but still under active development.  
The following improvements are planned for upcoming releases:

- **Error handling and validation:**  
  Improve robustness by detecting invalid configurations and numerical instabilities.
- **Unit testing:**  
  Add automated tests to ensure correctness, stability, and reproducibility.

## 🧩 Overview

The Processor is Jacinta’s internal stochastic decision core.  
It maintains a multivariate Gaussian distribution over an internal vector `y` and updates this distribution based solely on scalar reward feedback.

Concretely, the Processor models:

```math
y \sim \mathcal{N}(\mu, \Sigma)
```

On each interaction cycle, it can:

- Sample a new candidate `y` from the current distribution.
- Receive a scalar reward `r` associated with that sample.
- Update its parameters `(μ, Σ)` to favor better-performing regions of the space.

This makes the Processor suitable as a lightweight engine for black-box optimization, stochastic policy search, or exploratory control in continuous spaces.

## 🎯 Purpose

The Processor’s purpose is to represent and refine an internal probability distribution over “good behaviors”.

In particular, it:

- Keeps track of a mean `μ` (current best guess) and covariance `Σ` (current uncertainty and correlation structure).
- Draws samples from this distribution to generate new candidates.
- Uses scalar rewards to decide whether to move `μ` and/or reshape `Σ` around those samples.
- Normalizes rewards so that learning remains stable across different reward scales.

The end result is a mechanism that learns where to concentrate probability mass (exploitation) while still maintaining exploration where the environment suggests potential gains.

## ⚙️ Design Philosophy

The Processor is built around adaptive stochastic search with a focus on stability and simplicity.

- **Stochastic by design:**  
  Sampling from a Gaussian naturally introduces exploration; the covariance `Σ` controls both the scale and the directions of that exploration.

- **Self-normalizing rewards:**  
  Rewards are centered around a running baseline and scaled by a running estimate of their magnitude. This keeps update steps roughly comparable even if raw rewards drift or are noisy.

- **Gradient-free learning:**  
  The Processor does not require gradients from the environment. It only needs scalar rewards, making it usable in black-box scenarios.

- **Covariance regularization:**  
  The covariance matrix is kept symmetric and its diagonal elements are clamped to a minimum variance `min_var`. This encourages positive-definite, well-conditioned matrices suitable for Cholesky factorization.

- **Dynamic dimensionality:**  
  Dimensions can be added or removed at runtime. New dimensions are introduced as independent components with configurable mean and variance.

- **Serializable state:**  
  All key parameters, learning rates, and reward-normalization buffers can be saved and restored as JSON via `to_dict` / `from_dict` and `save` / `load`.

- **Adaptive variance and directional learning:**  
  The Processor adjusts both the mean `μ` and covariance `Σ` based on two aspects of each sample:

  1. how good or bad its reward is relative to the current baseline, and
  2. how common or rare that sample is under the current Gaussian.

  Intuitively:

  - **High rewards for common samples:**  
    → The mean `μ` moves slightly toward those values.  
    → The covariance `Σ` contracts (variance decreases).  
    → Interpretation: “What we usually do is working well; be more confident and explore less around here.”

  - **High rewards for rare or unusual samples:**  
    → The mean `μ` moves toward those atypical values.  
    → The covariance `Σ` expands (variance increases).  
    → Interpretation: “An unusual behavior performed surprisingly well; explore more in this direction.”

  - **Low rewards for common samples:**  
    → The mean `μ` moves away from those values.  
    → The covariance `Σ` expands (variance increases).  
    → Interpretation: “What we typically do is not good enough; increase exploration to find alternatives.”

  - **Low rewards for rare or unusual samples:**  
    → The mean `μ` moves away from those sampled points.  
    → The covariance `Σ` contracts (variance decreases).  
    → Interpretation: “That rare deviation did not pay off; reduce exploration in that direction.”

  This pattern emerges from how the update rule reshapes `Σ` around `y - μ`, and provides a natural balance between confidence (contraction) and curiosity (expansion).

## 🧠 Mathematical Transformation

The Processor operates as a closed adaptive loop based on the cycle  
sampling → reward evaluation → parameter adaptation → regularization.  
Each stage follows well-defined mathematical rules that together form a stable learning process.

### 1. Sampling — Generating Internal Candidates

The Processor models its internal behavior as a multivariate Gaussian:

```math
y \sim \mathcal{N}(\mu, \Sigma)
```

New samples are drawn using the reparameterization form:

```math
z \sim \mathcal{N}(0, I)
```

```math
\Sigma = L L^T
```

```math
y = \mu + L z
```

This produces a sample `y` distributed according to the current belief about optimal behavior.

### 2. Reward Normalization — Centering and Scaling

Since rewards may vary in scale and distribution, the Processor maintains  
two exponential moving averages to standardize them: a baseline and a scale.

#### Baseline update

```math
r_{baseline} \leftarrow (1 - \alpha) r_{baseline} + \alpha r
```

The baseline represents the expected reward, acting as a dynamic center.

#### Reward centering

```math
r_{centered} = r - r_{baseline}
```

This expresses how much better or worse the current reward is compared to the moving average.

#### Scale update

```math
r_{scale} \leftarrow (1 - \beta) r_{scale} + \beta |r_{centered}|
```

The scale tracks the mean absolute deviation of rewards, allowing normalization across varying magnitudes.

#### Normalized reward

```math
\tilde{r} = \frac{r_{centered}}{r_{scale} + \epsilon}
```

This normalized reward $ \tilde{r} $ becomes the driving signal for learning,  
ensuring consistent adaptation even if raw rewards drift over time.

### 3. Mean Update — Adjusting Central Expectation

The mean vector `μ` defines the Processor’s current expectation of optimal outcomes.  
Its update rule determines how the distribution shifts toward or away from recent samples.

Let the sample deviation be:

```math
d = y - \mu
```

The update direction is approximated as:

```math
v_\mu = \Sigma^{-1} d
```

and normalized to unit length:

```math
v_\mu = \frac{v_\mu}{\|v_\mu\| + \epsilon}
```

The mean is then updated proportionally to the normalized reward:

```math
\mu \leftarrow \mu + \eta_\mu	\tilde{r} v_\mu
```

A positive normalized reward moves `μ` toward the sampled point (`y`);  
a negative reward moves it away.

### 4. Covariance Update — Shaping Exploration and Confidence

The covariance matrix `Σ` governs the Processor’s uncertainty.  
It expands or contracts depending on how novel and rewarding each sample is.

First, the residual difference between the sample covariance and the model covariance is computed:

```math
M = d d^T - \Sigma
```

This captures whether the observed deviation `d` suggests more or less variability than expected.

The covariance is updated proportionally to this residual and the normalized reward:

```math
\Sigma \leftarrow \Sigma + \eta_\Sigma \tilde{r} \frac{M}{\|M\|_F + \epsilon}
```

This rule produces the following emergent behaviors:

| Reward Type | Sample Type | Effect on `μ`    | Effect on `Σ` | Interpretation                            |
| ----------- | ----------- | ---------------- | ------------- | ----------------------------------------- |
| High        | Common      | Toward sample    | Contracts     | Confirms stable good behavior             |
| High        | Rare        | Toward sample    | Expands       | Encourages exploration around new success |
| Low         | Common      | Away from sample | Expands       | Seeks alternatives to habitual failure    |
| Low         | Rare        | Away from sample | Contracts     | Avoids unproductive deviations            |

This balance ensures the Processor increases confidence in consistently good behaviors while remaining open to exploration when performance drops.

### 5. Regularization — Maintaining Valid Covariance

To ensure numerical stability and valid probabilistic interpretation,  
the Processor applies two regularization operations after each update:

#### Symmetrization

```math
\Sigma \leftarrow \frac{1}{2}(\Sigma + \Sigma^T)
```

This enforces perfect symmetry, compensating for small numerical asymmetries.

#### Variance floor

```math
\Sigma_{ii} \leftarrow \max(\Sigma_{ii}, \text{min\_var})
```

This prevents the covariance matrix from degenerating (e.g., singular or negative variances),  
maintaining it positive definite and suitable for Cholesky decomposition.

## 🧱 Class Interface

### `Processor(size, mu=0.0, sigma=100.0, lr_mu=1e-3, lr_sigma=1e-4, min_var=1e-8, r_alpha=1e-2, r_beta=1e-2, eps=1e-12)`

**Parameters**

- `size (int)`  
  Dimensionality of the process.
- `mu (float | np.ndarray)`  
  Initial mean.
  - If scalar, it is broadcast to all dimensions.
  - If array, it is copied as given.
- `sigma (float | np.ndarray)`
  - If scalar, interpreted as a standard deviation and expanded to `sigma**2 * I`.
  - If array, used as a full covariance matrix.
- `lr_mu (float)`  
  Learning rate for the mean update.
- `lr_sigma (float)`  
  Learning rate for the covariance update.
- `min_var (float)`  
  Minimum allowed variance per dimension, used as a stability floor.
- `r_alpha (float)`  
  EMA factor for the reward baseline.
- `r_beta (float)`  
  EMA factor for the reward scale.
- `eps (float)`  
  Small numerical constant used to avoid division by zero.

### Core Methods

| Method                               | Description                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `process_forward()`                  | Samples a vector `y` from the current Gaussian `N(μ, Σ)` and caches it. |
| `process_backward(r)`                | Updates `μ` and `Σ` using the reward `r` for the last sampled `y`.      |
| `add_dimension(mu=0.0, sigma=100.0)` | Adds a new independent dimension with the given mean and std.           |
| `remove_dimension(idx)`              | Removes a dimension (both from `μ` and `Σ`) by index.                   |
| `copy()`                             | Returns a deep copy of the Processor and its internal state.            |
| `to_dict()` / `from_dict()`          | Serialize / deserialize Processor configuration and learning state.     |
| `save(path)` / `load(path)`          | Save or load Processor state as JSON.                                   |
| `N` _(property)_                     | Returns the current number of dimensions.                               |

## 🧩 Example Usage

```python
import numpy as np

from jacinta.nodes import Processor


# Initialize a 3D Processor with moderate initial variance
p = Processor(size=3, mu=0.0, sigma=5.0)

for step in range(10):
    # 1. Sample from the current internal distribution
    y = p.process_forward()

    # 2. Evaluate the sample with some external reward function
    #    Example: we prefer values close to the origin
    reward = -np.linalg.norm(y)

    # 3. Update Processor parameters based on the observed reward
    p.process_backward(reward)

    print(f"Step {step:02d} | y = {y}, reward = {reward:.3f}")
```

**Possible Output**

```
Step 00 | y = [1.22378151 0.94937674 1.61261365], reward = -2.236
Step 01 | y = [-9.98153573 -2.38306249 -5.30725803], reward = -11.553
Step 02 | y = [-4.65634864 -4.05203677  2.03677565], reward = -6.500
...
```

Over time, the mean vector `μ` tends to move toward high-reward regions (here, near the origin), while the covariance `Σ` dynamically contracts or expands to reflect Jacinta’s current balance between confidence and exploration.
