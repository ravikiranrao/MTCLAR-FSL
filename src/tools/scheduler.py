import torch

def get_scheduler(optimizer, name, cfg, steps_per_epoch=None, epochs=None):
    assert name in ["StepLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "OneCycleLR"]
    if name == "StepLR":
        assert cfg["STEP_SIZE"] is not None
        assert cfg["GAMMA"] is not None
        # Decrease learning rate by a factor of 10 every 15 epochs.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["STEP_SIZE"], gamma=cfg["GAMMA"])

    elif name == "CosineAnnealingWarmRestarts":
        assert cfg["T_0"] is not None
        assert cfg["T_mult"] is not None
        assert cfg["eta_min"] is not None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"],
                                                                         T_mult=cfg["T_mult"],
                                                                         eta_min=cfg["eta_min"])
    elif name == "ReduceLROnPlateau":
        assert cfg["MODE"] is not None
        assert cfg["FACTOR"] is not None
        assert cfg["PATIENCE"] is not None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg["MODE"], factor=cfg["FACTOR"],
                                                               patience=cfg["PATIENCE"])

    elif name == "OneCycleLR":
        assert cfg["max_lr"] is not None
        assert steps_per_epoch is not None
        assert epochs is not None

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
                                                        steps_per_epoch=steps_per_epoch,
                                                        epochs=epochs)
    else:
        raise ValueError

    return scheduler
