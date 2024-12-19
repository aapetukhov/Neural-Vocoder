import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer, Inferencer
from src.utils.io_utils import ROOT_PATH
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

import torch._dynamo
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesise")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    generator = instantiate(config.generator).to(device)
    
    metrics = {"inference": []}
    for metric_config in []:
        metrics["inference"].append(instantiate(metric_config))

    save_path = ROOT_PATH / "saved_audios" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        save_path=save_path,
        metrics=metrics,
        batch_transforms=batch_transforms
    )

    part_logs = inferencer.run_inference()
    for part_name, logs in part_logs.items():
        for k, v in logs.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()