"""https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch"""
import pytorch_lightning as pl


class InputMonitor(pl.Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram(
                "target", y, global_step=trainer.global_step
            )
