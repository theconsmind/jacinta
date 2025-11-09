# Transmitter Module

## 🚧 Roadmap / Next Steps

The Transmitter module is operational but still evolving.  
The following improvements are planned for future versions:

- **Error handling and validation:**  
  Add safeguards for inconsistent bounds or invalid inputs.
- **Unit testing:**  
  Introduce comprehensive tests to verify numerical stability and correct inverse behavior with respect to the Receiver.
- **Configurable transformation function:**  
  Allow users to define custom "squashing" functions (e.g., `tanh`, `sigmoid`, or custom mappings) instead of being limited to the hyperbolic tangent.

## 🧩 Overview

The Transmitter module is the final stage in Jacinta’s core pipeline.  
It takes internal outputs — potentially unbounded — and converts them into an external bounded representation suitable for interaction with the environment.

Where the Receiver converts finite intervals into ℝ, the Transmitter does the reverse:  
it compresses real-valued signals back into their natural or physical limits.

## 🎯 Purpose

Transmitter’s role is to ensure that Jacinta’s outputs are:

- Interpretable by the external world (e.g., control commands, actuator ranges, probability scales).
- Numerically stable, by smoothly bounding values within `[min_x, max_x]`.
- Symmetric with respect to the Receiver, maintaining consistent forward–inverse mapping.

In essence, the Transmitter performs a controlled “compression” of Jacinta’s raw internal activations into valid, externally usable data.

## ⚙️ Design Philosophy

The Transmitter was built with the following guiding principles:

- **Inversion symmetry:**  
  It mirrors the Receiver’s transformation logic exactly, ensuring that round trips (Receiver → Processor → Transmitter) are mathematically consistent.

- **Explicit bounds:**  
  Each dimension’s limits are clearly defined; unbounded or partially bounded dimensions remain untouched.

- **Smooth transitions:**  
  Uses `tanh` to smoothly map from ℝ → (-1, 1), avoiding hard clipping or discontinuities.

- **Numerical safety:**  
  Operations are fully vectorized and rely on NumPy’s stable implementations of `tanh`.

- **Dynamic adaptability:**  
  The number of output dimensions and their bounds can change at runtime.

- **Serializable state:**  
  Full configuration can be saved, loaded, or cloned to ensure reproducibility in simulations and AI deployments.

## 🧠 Mathematical Transformation

For each dimension _i_ with both finite bounds:

1. **Squash to (-1, 1):**

   ```math
   z_i = \tanh(x_i)
   ```

2. **Rescale to [0, 1]:**

   ```math
   u_i = \frac{z_i + 1}{2}
   ```

3. **Map to [min, max]:**

   ```math
   y_i = \text{min}_i + u_i (\text{max}_i - \text{min}_i)
   ```

Unbounded or partially bounded dimensions remain unchanged.

This process is the exact mathematical inverse of the Receiver’s transformation, guaranteeing continuity and reversibility.

## 🧱 Class Interface

### `Transmitter(size, min_x=None, max_x=None)`

**Parameters**

- `size (int)`  
  Number of output dimensions.

- `min_x (float | np.ndarray | None)`  
  Lower bounds per dimension (`np.nan` or `None` = unbounded).

- `max_x (float | np.ndarray | None)`  
  Upper bounds per dimension (`np.nan` or `None` = unbounded).

### Core Methods

| Method                                  | Description                                                   |
| --------------------------------------- | ------------------------------------------------------------- |
| `process_forward(x)`                    | Maps internal unbounded signal `x` to external bounded space. |
| `add_dimension(min_x=None, max_x=None)` | Adds a new output dimension with optional bounds.             |
| `remove_dimension(idx)`                 | Removes a dimension and its bounds by index.                  |
| `copy()`                                | Deep copy preserving configuration.                           |
| `to_dict()` / `from_dict()`             | Serialize / deserialize configuration.                        |
| `save(path)` / `load(path)`             | Save or restore configuration as JSON.                        |
| `N` _(property)_                        | Returns the number of dimensions.                             |

## 🧩 Example Usage

```python
import numpy as np

from jacinta.nodes import Transmitter


# Define a Transmitter with mixed bounds:
#  - Dimension 0: fully bounded [0, 1]
#  - Dimension 1: unbounded
#  - Dimension 2: only lower bound (>= 0)
#  - Dimension 3: only upper bound (<= 10)
t = Transmitter(
    size=4,
    min_x=[0, np.nan, 0, np.nan],
    max_x=[1, np.nan, np.nan, 10]
)

# Example internal signal (output from Processor)
x = np.array([0.0, 5.0, -2.0, 3.0])

# Apply the Transmitter transformation
y = t.process_forward(x)

print("Internal x:", x)
print("External y:", y)
```

**Output**

```
Internal x: [ 0.  5. -2.  3.]
External y: [ 0.5  5.  -2.   3. ]
```

Here:

- The first value is smoothly mapped into `[0, 1]`.
- Unbounded or partially bounded values pass through unchanged.
- The transformation ensures valid external ranges without sharp clipping.
