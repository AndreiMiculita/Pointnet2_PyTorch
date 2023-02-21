import hashlib
import os
import time

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))

    # generate a unique hash to save checkpoints, based on the checksum of the starting time
    # this is useful to avoid overwriting checkpoints when running multiple experiments

    exp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[0:8]

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filepath=os.path.join(
            cfg.task_model.name, exp_hash + "-{epoch}-{val_loss:.4f}"
        ),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        logger=TensorBoardLogger("tb_logs", name=cfg.task_model.name),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
