## Structure of this Repo

- [src](https://github.com/Laggg/neural_env_surviv/blob/main/src) : main env engine code
- [demo_app](https://github.com/Laggg/neural_env_surviv/blob/main/demo_app) : playable app
- [scripts](https://github.com/Laggg/neural_env_surviv/blob/main/scripts) : scripts for data downloading
- data : downloaded data will be saved here
- weights : NN weights will be save here

## Requirements

- python3
- pip

## Installation & Run

### Clone the repo and change to the project root directory:

```
git clone https://github.com/Laggg/neural_env_surviv.git
cd neural_env_surviv
```


### Create venv:

via `conda`:

```
conda create -n survivio_venv python=3.8
conda activate survivio_venv
```

via `python`:

```
// Linux:
python -m venv survivio_venv
source survivio_venv/bin/activate
```

```
// Windows:
python -m venv survivio_venv
source survivio_venv/Scripts/activate
```

### Install requirements:

if you have `CPU`:
```
python -m pip install -r requirements-cpu.txt
```

if you have `CUDA` (haven't tested on Linux so **not recommended**):
```
python -m pip install -r requirements-gpu.txt
```

### And run:

```
python startup.py
```

You should see a window. Play with `wasd`. To close game - press `e`. 
