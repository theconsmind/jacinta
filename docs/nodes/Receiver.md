# Receiver Module

## 🚧 Roadmap / Next Steps

The Receiver module is functional but still under active development.  
The following features are planned for upcoming releases:

- **Error handling and validation:**  
  Improve robustness by detecting invalid configurations and runtime inconsistencies.
- **Unit testing:**  
  Add automated test coverage to ensure numerical stability and backward compatibility.
- **Configurable transformation function:**  
  Allow different unbounding strategies (e.g., `tanh⁻¹`, `logit`, or custom mappings) instead of being limited to the hyperbolic tangent.

## 🧩 Overview

The **Receiver** module is the entry point of Jacinta’s data pipeline.  
It handles external inputs and transforms them into a standardized internal representation, optionally mapping bounded components into the full real line (ℝ).

This makes Receiver especially useful when dealing with real-world data that has natural limits (e.g., normalized features, sensor ranges, physical constraints).

## 🎯 Purpose

Receiver’s mission is to make external data **ready for internal processing**.  
In particular, it:

- Accepts data of arbitrary dimensionality.
- Optionally rescales components bounded in `[min_x, max_x]` into an **unbounded space** using a stable mathematical transform.
- Leaves unbounded or partially bounded dimensions untouched.
- Preserves the structure and ordering of the data.

This helps ensure that later stages in Jacinta (e.g., the `Processor`) operate in a continuous, smooth domain without artificial clipping or discontinuities.

## ⚙️ Design Philosophy

Receiver is designed with **clarity, safety, and reversibility** in mind.

- **Explicit bounds:**  
  Each dimension explicitly declares whether it is bounded, partially bounded, or unbounded. No implicit assumptions.

- **Selective normalization:**  
  Only dimensions with _both_ finite bounds are normalized.  
  Others are transparently passed through.

- **Numerical stability:**  
  A small `ε` margin prevents hitting singularities (e.g., `artanh(±1)`), ensuring smooth gradients and finite outputs.

- **Reversibility by design:**  
  The transformation is bijective for bounded dimensions, meaning it can be inverted perfectly by the corresponding Transmitter module.

- **Extendability:**  
  The number of dimensions and their bounds can be modified at runtime, making the Receiver adaptable to dynamic input configurations.

- **Serialization:**  
  Every instance can be saved, loaded, and cloned, ensuring reproducibility in complex AI pipelines.

## 🧠 Mathematical Transformation

For each dimension _i_ that has both bounds defined:

1. **Normalize to [0, 1]:**

   ```math
   u_i = \frac{x_i - \min_i}{\max_i - \min_i}
   ```

2. **Shift to [-1, 1]:**

   ```math
   v_i = 2u_i - 1
   ```

3. **Map to ℝ (unbound transformation):**

   ```math
   y_i = \tfrac{1}{2} \ln\!\left(\frac{1 + v_i}{1 - v_i}\right)
        = \mathrm{artanh}(v_i)
   ```

This transformation preserves continuity and order:  
values near the midpoint of the interval map to 0, while approaching the bounds results in large positive or negative outputs.

Dimensions with undefined or infinite bounds remain unchanged.

## 🧱 Class Interface

### `Receiver(size, min_x=None, max_x=None, eps=1e-12)`

**Parameters**

- `size (int)`: Dimensionality of the input vector.
- `min_x (float | np.ndarray | None)`: Lower bounds per dimension; use `np.nan` or `None` for unbounded.
- `max_x (float | np.ndarray | None)`: Upper bounds per dimension; use `np.nan` or `None` for unbounded.
- `eps (float)`: Small positive constant used for numerical stability.

### Core Methods

| Method                                  | Description                                                             |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `process_forward(x)`                    | Transforms input vector `x` into its unbounded internal representation. |
| `add_dimension(min_x=None, max_x=None)` | Dynamically appends a new dimension with optional bounds.               |
| `remove_dimension(idx)`                 | Removes a dimension and its associated bounds by index.                 |
| `copy()`                                | Deep copy preserving configuration.                                     |
| `to_dict()` / `from_dict()`             | Serialize / deserialize configuration.                                  |
| `save(path)` / `load(path)`             | Save or restore configuration as JSON.                                  |
| `N` _(property)_                        | Returns the number of dimensions.                                       |

## 🧩 Example Usage

```python
from jacinta.nodes import Receiver
import numpy as np

# Define a Receiver with mixed bounds:
#  - Dimension 0: fully bounded [0, 1]
#  - Dimension 1: unbounded (both sides)
#  - Dimension 2: only lower bound (>= 0)
#  - Dimension 3: only upper bound (<= 5)
r = Receiver(
    size=4,
    min_x=[0, np.nan, 0, np.nan],
    max_x=[1, np.nan, np.nan, 5]
)

# Example input vector
x = np.array([0.5, -3.2, 4.0, 2.5])

# Apply Receiver transformation
y = r.process_forward(x)

print("Input x:", x)
print("Transformed y:", y)
```

**Output**

```
Input x: [ 0.5 -3.2  4.   2.5]
Transformed y: [ 0.  -3.2  4.   2.5]
```

Here, only the first dimension is transformed (since it’s fully bounded),  
while the others remain unchanged because they lack one or both finite limits.
