import torch


class MaxLoss(torch.nn.Module):
    def forward(self, x, y):
        return torch.max(torch.abs(x - y))


def get_loss(loss, loss_reduction="mean"):
    if loss == "mse":
        return torch.nn.MSELoss(reduction=loss_reduction)
    elif loss == "mae":
        return torch.nn.L1Loss(reduction=loss_reduction)
    elif loss == "huber":
        return torch.nn.HuberLoss(reduction=loss_reduction)
    elif loss == "max":
        return MaxLoss()
    elif loss == "binary_cross_entropy":
        return torch.nn.BCEWithLogitsLoss(reduction=loss_reduction)
    else:
        raise NotImplementedError