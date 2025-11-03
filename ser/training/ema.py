from torch.optim.swa_utils import AveragedModel


def build_ema(model, decay: float):
    """
    torch.optim.swa_utils.AveragedModel を用いた EMA モデルの生成。
    """

    def ema_avg(ema_param, param, num_averaged):
        if num_averaged == 0:
            return param.detach()
        return decay * ema_param + (1.0 - decay) * param

    return AveragedModel(model, avg_fn=ema_avg)
