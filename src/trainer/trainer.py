import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.gen_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

        # TODO: refactor so it's more compact
        gen_outputs = self.generator(**batch)
        batch.update(gen_outputs)

        # discriminator outputs + step
        disc_outputs = self.discriminator(**batch, detach=True)
        batch.update(disc_outputs)

        all_disc_losses = self.disc_criterion(**batch)
        batch.update(all_disc_losses)

        if self.is_train:
            batch["disc_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.disc_optimizer.step()
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()

        disc_outputs = self.discriminator(**batch, detach=False)
        batch.update(disc_outputs)

        # generator outputs + step
        all_losses = self.gen_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["gen_loss"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
        else:
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_audio(self, audio, audio_name):
        audio = (
            audio / torch.max(torch.abs(audio)).detach().cpu()
        )
        self.writer.add_audio(
            audio_name,
            audio.float(),
            sample_rate=self.mel_spectrogram.sample_rate,
        )

    def log_spectrogram(self, spectrogram, output_audio, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        melspec = self.melspec(output_audio)[0].detach().cpu()

        self.writer.add_image("target", plot_spectrogram(spectrogram_for_plot))
        self.writer.add_image("pred", plot_spectrogram(melspec))

    def log_predictions(self, output_audio, audio, examples_to_log=1, **batch):
        for i, (pred, target) in enumerate(zip(output_audio, audio)):
            self.log_audio(target, f"gt_{i}")
            self.log_audio(pred, f"pred_{i}")
