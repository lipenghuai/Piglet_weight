# Soft-MoE extraction for YOLO26

This directory contains a portable Soft-MoE module package for YOLO26. It is
intended to be copied into a YOLO26 tree and registered as normal model YAML
modules.

## Contents

- `modules.py`: MoE blocks and backward-compatible aliases.
- `experts.py`, `routers.py`, `utils.py`, `loss.py`: core runtime pieces.
- `__init__.py`: public exports plus the small YOLO auxiliary-loss helper.

The extracted package intentionally excludes optional diagnostics, pruning, and
test scripts so it stays small and easy to copy into official YOLO26 code.

## Quick Check

```bash
python scripts/regression/soft_moe/test.py
```

## Standalone usage

Add `scripts/regression` to `PYTHONPATH`, then:

```python
import torch
from soft_moe import ModularRouterExpertMoE

x = torch.randn(2, 256, 40, 40)
layer = ModularRouterExpertMoE(256, 256, num_experts=4, top_k=2)
y = layer(x)
```

## YOLO26 integration checklist

1. Copy this folder into the target YOLO26 tree:

```text
<yolo26>/ultralytics/nn/modules/soft_moe
```

2. Export the modules from `<yolo26>/ultralytics/nn/modules/__init__.py`:

```python
from .soft_moe import (
    OptimizedMOE,
    OptimizedMOEImproved,
    MOE,
    ES_MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    A2C2fMoE,
    ABlockMoE,
)
```

Also add the same names to `__all__` if that file maintains one.

3. Import the same names in `<yolo26>/ultralytics/nn/tasks.py`, then add them
to `base_modules` inside `parse_model`. Add `A2C2fMoE` to `repeat_modules`.

For the parse branch, the same pattern as Conv/C2f-like modules is expected:

```python
if m in base_modules:
    c1, c2 = ch[f], args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
    if m in repeat_modules:
        args.insert(2, n)
        n = 1
    if m is A2C2fMoE:
        legacy = False
```

4. Use the layer in a model YAML:

```yaml
- [-1, 1, ModularRouterExpertMoE, [512, 4, 2]]
```

With the parser above this becomes:

```python
ModularRouterExpertMoE(in_channels, 512, num_experts=4, top_k=2)
```

5. Optional but recommended for training: sum auxiliary MoE losses in YOLO's
task loss. `soft_moe` exports `collect_moe_aux_loss(model)`.

Example loss hook:

```python
from ultralytics.nn.modules.soft_moe import collect_moe_aux_loss

moe_loss = collect_moe_aux_loss(self.model, self.device)
loss[3] = moe_loss * self.hyp.moe
```

If the official YOLO26 loss only returns three detection terms, allocate a
fourth slot or fold the MoE term into the total loss while still logging it.
When returning a fourth loss item, update the trainer loss names as well, for
example `("box_loss", "cls_loss", "dfl_loss", "moe_loss")`.

## Notes

- `ABlockMoE` and `A2C2fMoE` require `ultralytics.nn.modules.block`. The package
  can still be imported outside YOLO; those two classes raise a clear error only
  when instantiated without the YOLO block dependency.
- For inference-only integration, registering the modules and YAML entries is
  enough. Auxiliary loss wiring is only needed for training with load balancing.
