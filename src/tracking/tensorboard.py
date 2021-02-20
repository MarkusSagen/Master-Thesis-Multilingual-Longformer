from torch.utils.tensorboard import writer


def SummaryWriter(experiment: str) -> None:
    return writer.SummaryWriter(log_dir=f'/workspace/logs/{experiment}')
