# JAX-Trainer Example

This is an example of how to use the [JAX-Trainer library](https://github.com/phlippe/jax_trainer) to build a simple template for research projects.

## Installation

Install the library by cloning the [jax-trainer repository](https://github.com/phlippe/jax_trainer) and install it with `pip install -e .`. All dependencies will be installed automatically.

## Usage

The main entry point is the `main.py` file. All experiments can be run via:

```bash
python main.py
```

Different experiments and/or models can be selected by creating new configs in the subfolder `cfg`, or by overwriting the values of an existing config with command-line arguments. In this template, the default experiments is a simple CNN trained on CIFAR10 classification. To train it with a different learning rate, run:

```bash
python main.py --cfg.optimizer.lr 0.004
```

Another experiment is an autoencoder on the CIFAR100 dataset, which is specified in the config `cfg/autoencoder.yaml`. To use this config, run:

```bash
python main.py --cfg cfg/default_config.py:autoencoder
```

## Project Structure

The project is structured into the following folders:

- `cfg`: This folder contains all config files. Configs are specified in YAML format and can be nested. The default config is `cfg/default_config.yaml`, and is parsed with `cfg/default_config.py`. Additional configs like `cfg/autoencoder.yaml` can be defined to run different experiments, using the same python file for parsing.
- `datasets`: This folder contains implementations of datasets and data utilities. For each dataset, a construction function like `build_my_dataset(dataset_config)` should be defined, which can be specified in the config to execute. The function should return a `jax_trainer.datasets.DatasetModule`, containing a train, val, and test dataset as well as their respective data loaders.
- `experiments`: This folder contains task-specific code, such as the trainer modules. Each trainer should inherit from `jax_trainer.trainer.TrainerModule`, and overwrite at least the loss function. The trainer module can then be specified in the config to execute. Additionally, task-specific logging functions and callbacks can be defined here.
- `models`: This folder contains implementations of models. The models can be arbitrary flax modules and should be specified in the config to execute. The hyperparameters in the config will then be passed to the model constructor.
- `main.py`: This is the main entry point of the project. It parses the config, sets the requested GPUs, builds the dataset and trainer, and finally executes the experiment.

### Implementing your own experiment

To implement your own experiment, you can follow the following steps:

1. If you want to train on a new dataset, create a dataset construction function in the `datasets` folder.
2. If you want to train a new model, create a model implementation in the `models` folder.
3. If you want to train a new task, create a trainer module in the `experiments` folder. If needed, you can also create a new callback or logging function.
4. Create a new config in the `cfg` folder, e.g. `cfg/my_config.yaml`, and specify the dataset, model, and trainer to use.
5. Run the experiment with `python main.py --cfg cfg/default_config.py:my_config`.
