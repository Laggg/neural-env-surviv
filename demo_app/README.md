## Structure of this Repo

- [src](src) : main env engine code
- [demo_app](demo_app) : playable app
- [scripts](scripts) : scripts for data downloading
- [data](data) : downloaded data saved here
- [weights](weights) : NN weights

## Requirements

- python3
- pip

## Installation & Run
### From source

Clone the repo and change to the project root directory:

```
git clone https://github.com/Laggg/neural_env_surviv.git
cd neural_env_surviv
```

1. Create venv:

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

2. Install requirements:

   if you have `GPU`:
   ```
   python -m pip install -r requirements-gpu.txt
   ```
   
   if you have `CPU` only:
   ```
   python -m pip install -r requirements-cpu.txt
   ```

3. And run:

   ```
   python startup.py
   ```
   
   You should see a window. Play with `wasd`. To close game - press `e`. 
