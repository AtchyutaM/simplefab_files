# SimpleFab

A small discrete-time semiconductor fab toy model (4-machine line, 2 products) with:
- deterministic scenario generation (`make_common_config`)
- event-driven simulation (`ProductionLine`)
- optional Gantt plotting
- Gymnasium environment wrapper (`FabEnv`)
- PPO training script (Stable-Baselines3)

## Quickstart (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run a single simulation (with plots + CSV)

```powershell
python scripts\run_sim.py
```

Outputs:
- `output_data/final_event_log.csv`
- queue plots + an optional Gantt plot window

### Train PPO (TensorBoard)

```powershell
python scripts\train_ppo.py
tensorboard --logdir .\tb_fab --reload_interval 2
```

## Package layout

```
simplefab/
  simplefab/
    config.py
    sim.py
    gantt.py
    env.py
    eval.py
  scripts/
    run_sim.py
    train_ppo.py
```
