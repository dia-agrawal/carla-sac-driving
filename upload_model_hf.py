"""
Upload the trialI3000s SAC checkpoint to Hugging Face Hub.

Usage:
    huggingface-cli login          # one-time login
    python upload_model_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "dia-agrawal/carla-sac-driving"
MODEL_DIR = os.path.join("tmp", "sac", "trialI3000s")
PLOT = os.path.join("plots", "trialI3000s.png")

MODEL_CARD = """---
license: mit
tags:
  - reinforcement-learning
  - carla
  - autonomous-driving
  - sac
  - pytorch
---

# CARLA SAC Driving — trialI3000s

Soft Actor-Critic (SAC) agent trained for autonomous driving in the
[CARLA simulator](https://carla.org/) (v0.9.16).

## Training

- **Steps:** 3000 training episodes
- **Algorithm:** SAC with automatic entropy tuning (PyTorch)
- **Environment:** Custom Gymnasium wrapper around CARLA 0.9.16
- **Observation:** GPS position, heading, speed, route waypoints (Kalman-filtered)
- **Actions:** Continuous throttle/brake and steering

## Files

| File | Description |
|---|---|
| `actor_sac.pt` | Actor network weights |
| `critic_1_sac.pt` / `critic_2_sac.pt` | Twin Q-network weights |
| `target_critic_1_sac.pt` / `target_critic_2_sac.pt` | Target network weights |
| `alpha_sac.pt` | Learned entropy temperature |

## Usage

```python
import torch
import sys
sys.path.append("path/to/carla-sac-driving")

from SAC.networks import ActorNetwork

actor = ActorNetwork(alpha=9e-4, input_dims=(YOUR_OBS_DIM,),
                     max_action=1.0, n_actions=2)
actor.load_state_dict(torch.load("actor_sac.pt", map_location="cpu"))
actor.eval()
```

See the [GitHub repo](https://github.com/dia-agrawal/carla-sac-driving) for
full training code and environment setup.
"""

def main():
    api = HfApi()

    print(f"Creating repo {HF_REPO_ID} ...")
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

    # Write model card
    card_path = os.path.join(MODEL_DIR, "README.md")
    with open(card_path, "w") as f:
        f.write(MODEL_CARD)

    # Upload all checkpoint files
    print(f"Uploading files from {MODEL_DIR} ...")
    for fname in os.listdir(MODEL_DIR):
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  uploading {fname}")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=HF_REPO_ID,
                repo_type="model",
            )

    # Upload training plot if it exists
    if os.path.exists(PLOT):
        print(f"  uploading {os.path.basename(PLOT)}")
        api.upload_file(
            path_or_fileobj=PLOT,
            path_in_repo=os.path.basename(PLOT),
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    print(f"\nDone! Model uploaded to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
