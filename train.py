import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator, _convert_="partial").to(device)
    discriminator = instantiate(config.discriminator, _convert_="partial").to(device)
    logger.info(generator)
    logger.info(discriminator)

    # get function handles of loss and metrics
    gen_loss_function = instantiate(config.gen_loss_function).to(device)
    disc_loss_function = instantiate(config.disc_loss_function).to(device)
    
    metrics = {"train": [], "inference": []}

    # build optimizer, learning rate scheduler for gen
    gen_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = instantiate(config.optimizer, params=gen_params)
    gen_scheduler = instantiate(config.gen_scheduler, optimizer=gen_optimizer)

    # build optimizer, learning rate scheduler for disc
    disc_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    disc_optimizer = instantiate(config.optimizer, params=disc_params)
    disc_scheduler = instantiate(config.disc_scheduler, optimizer=disc_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        gen_criterion=gen_loss_function,
        disc_criterion=disc_loss_function,
        metrics=metrics,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        gen_scheduler=gen_scheduler,
        disc_scheduler=disc_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
