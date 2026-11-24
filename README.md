# endfire-sim

Simulation of acoustic waves, microphone directivity, and endfire arrays in space and time.
Sampled spherical-source propagation is **powered by [Acoular](https://acoular.org/)**.

<a href="https://acoular.org/"><img src="https://github.com/acoular.png?size=80" width="40" alt="Acoular"></a>

## Install

With pip from GitHub:

```bash
python -m pip install git+https://github.com/jake-is-ESD-protected/endfire-sim.git
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/jake-is-ESD-protected/endfire-sim.git
```

## Minimal example

Render a sampled spherical source at two microphone positions:

```python
import numpy as np
from endfiresim.wave import CWaveModelSpheric

fs = 48_000
source = CWaveModelSpheric("wn", amp=1.0, source_xyz=(0, 0, 0), duration=0.1)

positions = [(1, 0, 0), (2, 0, 0)]
field = source.field(positions, n_samples=int(0.1 * fs), fs=fs)
print(field.shape)  # (4800, 2)
```

Analytic planar and spherical waves, superposition, cardioid sensors, and endfire beamformers are also available. See `examples/` for notebooks.

## Development

```bash
uv run pytest
```

Released under the GPL-3.0-only license.
