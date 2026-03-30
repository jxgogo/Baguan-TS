# BaguanTS

BaguanTS is a time series forecasting model based on in-context learning with transformer architecture. It supports both 2D (tabular) and 3D (time series) inference modes.

## Envs

```bash
pip install -r requirements.txt
```

## Quick Start

### Inference Mode

```bash
python evaluate_BaguanTS.py
```

Configuration files:
- Model config: `./configs/model_config.yml`
- Hyperparameters: `./configs/hyper_config.yml` (3D mode) or `./configs/hyper_config_2d.yml` (2D mode)

### Usage Example

```python
from BaguanTS import BaguanTS

# Initialize model
model = BaguanTS(
    ckpt_path="path/to/checkpoint.ckpt",
    config_path="./configs/model_config.yml",
    device="cuda:0"
)

# Run prediction
forecast, forecast_quantiles = model.predict(
    X_train, y_train, X_test,
    data_type='TS-tabular',
    context_len=576,
    K=30
)
```

## Project Structure

```
.
├── BaguanTS.py              # Main inference interface
├── evaluate_BaguanTS.py     # Evaluation script with toy example
├── configs/                 # Configuration files
│   ├── model_config.yml     # Model architecture config
│   ├── hyper_config.yml     # 3D inference hyperparameters
│   └── hyper_config_2d.yml  # 2D inference hyperparameters
└── src/                     # Source code
    ├── base/                # Abstract base classes
    ├── modules/             # Model components (encoder, decoder, attention, etc.)
    ├── pipeline/            # Model factory and pipeline
    └── utils/               # Utility functions
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{baguants2026,
  title={BaguanTS: Time Series Forecasting with In-Context Learning},
  author={},
  journal={},
  year={2026}
}
```

## Note

Model checkpoints will be released upon paper publication.
