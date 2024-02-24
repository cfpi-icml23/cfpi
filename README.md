# cfpi

<div align="center">

[![Build status](https://github.com/cfpi-icml23/code/actions/workflows/build.yml/badge.svg)](https://github.com/ezhang7423/cfpi/actions?query=workflow%3Abuild)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/ezhang7423/cfpi/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/ezhang7423/cfpi/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/ezhang7423/cfpi/releases)
[![License](https://img.shields.io/github/license/ezhang7423/cfpi)](https://github.com/ezhang7423/cfpi/blob/master/LICENSE)

Offline Reinforcement Learning with Closed-Form Policy Improvement Operators

</div>

## ⚓ Installation

We require [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (a faster drop-in replacement of conda) or [conda](https://docs.conda.io/en/latest/miniconda.html). Mambaforge is recommended. To install, simply run

```bash
make install
```

If you'd like to install without downloading data, run 

```bash
make NO_DATA=1 install
```
You'll then need to install [mujoco 210](https://github.com/deepmind/mujoco/releases/tag/2.1.0) to ~/.mujoco/mujoco210/ and add the following to your `.bashrc`: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:$HOME/.mujoco/mjpro150/bin`.


## Get Started

```
# example run
cfpi bc

# help
cfpi --help

 Usage: cfpi [OPTIONS] ALGORITHM:{bc|mg|reverse_kl|sarsa_iqn|sg} VARIANT

╭─ Arguments ──────────────────────────────────────────────────────────────────────────╮
│ *    algorithm      ALGORITHM:{bc|mg|reverse_kl|sar  Specify algorithm to run. Find  │
│                     sa_iqn|sg}                       all supported algorithms in     │
│                                                      ./cfpi/variants/SUPPORTED_ALGO… │
│                                                      [default: None]                 │
│                                                      [required]                      │
│ *    variant        TEXT                             Specify which variant of the    │
│                                                      algorithm to run. Find all      │
│                                                      supported variant in            │
│                                                      ./cfpi/variants/<algorithm_nam… │
│                                                      [default: None]                 │
│                                                      [required]                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────╮
│ --parallel                          [AntMaze|MediumExpert|  Run multiple versions of │
│                                     MediumReplay|Single|Wi  the algorithm on         │
│                                     de]                     different environments   │
│                                                             and seeds.               │
│                                                             [default: None]          │
│ --gridsearch                        [CqlDeltas|Deltas|DetD  Do a gridsearch. Only    │
│                                     eltas|EasyBcq|Ensemble  supported when parallel  │
│                                     Size|FineGrainedDeltas  is also enabled          │
│                                     |Full|KFold|Mg|Pac|QTr  [default: None]          │
│                                     ainedEpochs|ReverseKl|                           │
│                                     Testing|TrqDelta]                                │
│ --dry                   --no-dry                            Just print the variant   │
│                                                             and pipeline.            │
│                                                             [default: no-dry]        │
│ --install-completion                                        Install completion for   │
│                                                             the current shell.       │
│ --show-completion                                           Show completion for the  │
│                                                             current shell, to copy   │
│                                                             it or customize the      │
│                                                             installation.            │
│ --help                                                      Show this message and    │
│                                                             exit.                    │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

Example command to run the Behavior Cloning `VanillaVariant` over Medium Expert datasets, gridsearching over Delta hyperparameters:
`cfpi bc --parallel MediumExpert  --gridsearch Deltas VanillaVariant`

## Note on variants

**Only variants that have a seed and env_id will run. Therefore, `Base*` are typically not runnable.** By default, if no variant is specified the vanilla variant will run.

## 🏗️ Development

1. Install `pre-commit` hooks:

```bash
make pre-commit-install
```

2. Run the codestyle:

```bash
make codestyle
```

## Debugging

If you get the error `Failed to unlock the collection!`, try running 

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

and rerun your command.


To enable or disable debug mode, copy and paste the `cfpi/conf_private.example.py` to `cfpi/conf_private.py`.

You can run the following helpful commands to breakpoint easily:

```
export PYTHONBREAKPOINT='IPython.core.debugger.set_trace
# OR
export PYTHONBREAKPOINT=ipdb.set_trace
```

Nodemon is another helpful tool. See example usage below:

```
nodemon -I -x 'cfpi bc --parallel MediumExpert  --gridsearch Deltas VanillaBCVariant' -e py
```

If the code randomly gets stuck, the mujoco lockfile may need to be deleted. You can do so by running the delmlock at `./scripts/delmlock`. We recommend adding this script to your path.

### Development features

- Support for `Python 3.8`.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/ezhang7423/cfpi/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/ezhang7423/cfpi/blob/master/setup.cfg).
- Automatic codestyle with [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade).
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with code-formatting.
- Type checks with [`mypy`](https://mypy.readthedocs.io); docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint); security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit)
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Ready-to-use [`.editorconfig`](https://github.com/ezhang7423/cfpi/blob/master/.editorconfig), [`.dockerignore`](https://github.com/ezhang7423/cfpi/blob/master/.dockerignore), and [`.gitignore`](https://github.com/ezhang7423/cfpi/blob/master/.gitignore). You don't have to worry about those things.

## Adding a new algorithm

To add a new algorithm, you need to do three things:

1. Create the algorithm and the respective experiment file in `./cfpi/algorithms`
2. Specify this algorithm in `./variants/SUPPORTED_ALGORITHMS.py`
3. Create a `./variants/<alg_name>/variant.py` file

### Makefile usage

[`Makefile`](https://github.com/ezhang7423/cfpi/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

<details>
<summary>4. Code security</summary>
<p>

```bash
make check-safety
```

This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.

```bash
make check-safety
```

</p>
</details>

</p>
</details>

<details>
<summary>5. Type checks</summary>
<p>

Run `mypy` static type checker

```bash
make mypy
```

</p>
</details>

<details>
<summary>6. Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>7. All linters</summary>
<p>

Of course there is a command to ~~rule~~ run all linters in one:

```bash
make lint
```

the same as:

```bash
make test && make check-codestyle && make mypy && make check-safety
```

</p>
</details>

<details>
<summary>8. Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/ezhang7423/cfpi/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>

### Poetry

Want to know more about Poetry? Check [its documentation](https://python-poetry.org/docs/).

<details>
<summary>Details about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy@latest`
- `poetry run pytest`
- `poetry publish --build`

etc

</p>
</details>

### Building and releasing

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish 🙂 `poetry publish --build`

## 🎯 What's next

- Add support for deterministic CFPI
- Add support for VAE-CFPI

## Lines of Code

![image](https://github.com/cfpi-icml23/cfpi/assets/54998055/21be945c-f271-49ea-99ca-822b5a3c58e4)

## 🛡 License

[![License](https://img.shields.io/github/license/ezhang7423/cfpi)](https://github.com/ezhang7423/cfpi/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/ezhang7423/cfpi/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{li2022offline,
    title={Offline Reinforcement Learning with Closed-Form Policy Improvement Operators},
    author={Jiachen Li and Edwin Zhang and Ming Yin and Qinxun Bai and Yu-Xiang Wang and William Yang Wang},
    journal={ICML},
    year={2023},
```

# 👏 Credits

This project would not be possible without the following wonderful prior work.

<a href="https://github.com/microsoft/oac-explore">Optimistic Actor Critic</a> gave inspiration to our
method,
<a href="https://github.com/Farama-Foundation/D4RL">D4RL</a>
provides the dataset and benchmark for evaluating the performance of our agent, and
<a href="https://github.com/rail-berkeley/rlkit/">RLkit</a> offered a strong RL framework
for building our code from.

Template: [`python-package-template`](https://github.com/TezRomacH/python-package-template)
