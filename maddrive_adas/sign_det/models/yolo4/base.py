from . losses import YoloBboxLoss

from torch import nn


class ImageProcessingModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def loss(self, y_pred, y_true):
        raise NotImplementedError('Must be implemented')


class BaseYolo4(ImageProcessingModel):
    def __init__(self, **config):
        super().__init__(**config)

        self.loss_obj = YoloBboxLoss(config)

    def loss(self, y_pred, y_true):
        return self.loss_obj(y_pred, y_true)
