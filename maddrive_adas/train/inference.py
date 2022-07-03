import torch
import numpy as np
import logging

from .utils import construct_model
from ..utils.nms import TorchNMS
from ..utils.torch_utils import set_half_precision
from ..utils.bbox import diou_xywh_torch

from concurrent.futures import ProcessPoolExecutor as PoolExecutor

logger = logging.getLogger(__name__)


class BaseInferExecutor:
    def __init__(self, model, device=None):
        self.model = model
        self.model_config = model.config

        logger.debug(f"Loading model with config: {self.model_config}")

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self):
        return self.model.__name__

    @classmethod
    def from_file(cls, model_filepath: str, **kwargs):
        """Create model from serialized file, it creates another instance so to sync use update_model_state()

        Args:
            model_filepath (str): model path to file in filesystem

        Returns:
            InferExecutor: class for inference
        """
        loaded_data = torch.load(model_filepath)
        config = loaded_data["config"]
        model_state = loaded_data["model_state_dict"]

        model_config = config["model"]

        model = cls.from_config(model_config=model_config, **kwargs)
        model.update_model_state(model_state)
        return model

    def update_model_state(self, model_state):
        self.model.load_state_dict(model_state)

    def infer_image(self, image):
        return self.infer_batch([image])[0]

    def infer_batch(self, images):
        raise NotImplementedError("Method not implemented")

    @staticmethod
    def process_prediction(preds, preproc_data, preproc_ops):
        for op in reversed(preproc_ops):
            preds = op.inverse_transform(preds=preds, data=preproc_data)

        return preds

    @staticmethod
    def preprocess_images(images, preproc_ops):
        batch_tensor = []

        # Letterboxing can be optimized with applying matrix operations (scale, pad)
        preproc_data = []
        for img in images:
            data = {}
            for op in preproc_ops:
                img, _ = op.transform(img, data=data)

            preproc_data.append(data)
            batch_tensor.append(img)

        batch_tensor = torch.stack(batch_tensor, axis=0)
        return batch_tensor, preproc_data


class InferExecutor(BaseInferExecutor):
    def __init__(
        self,
        model,
        device=None,
        nms_threshold=0.6,
        conf_threshold=0.4,
        use_soft_nms=False,
        use_half_precision=False,
        n_processes=1,
    ):
        super().__init__(model=model, device=device)

        logger.info(f"Using device {self.device}")

        self.n_processes = n_processes
        self.model_hw = self.model_config["infer_sz_hw"]
        self.use_half_precision = use_half_precision
        self.use_soft_nms = use_soft_nms

        if self.use_half_precision:
            set_half_precision(self.model)

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.labels = self.model_config["labels"]

        self._size_tnsr = (
            torch.FloatTensor(
                [
                    self.model_hw[1],
                    self.model_hw[0],
                    self.model_hw[1],
                    self.model_hw[0],
                ]
            )
            .view(1, 1, 4)
            .to(self.device)
        )

        self.preproc_ops = self.model_config["preprocessing"]
        self.nms = TorchNMS(iou=diou_xywh_torch, iou_threshold=nms_threshold)

    def map_labels(self, label_ids):
        return [self.labels[id_] for id_ in label_ids]

    @classmethod
    def from_config(cls, model_config: dict, **kwargs):
        """Create model from config, it creates another instance so to sync use update_model_state()

        Args:
            model_config (dict): model configuration

        Returns:
            InferExecutor: class for inference
        """
        model = construct_model(model_config, inference=True)
        return cls(model=model, **kwargs)

    def get_labels(self):
        return self.labels

    def infer_batch(self, imgs_list):
        batch_tensor, preproc_data = self.preprocess_images(imgs_list, self.preproc_ops)

        # NOTE - here we don`t apply shift and scale to all bboxes at once - just to demonstrate
        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device)
            if self.use_half_precision:
                batch_tensor = batch_tensor.half()

            outputs = self.model(batch_tensor)
            outputs[..., :4] *= self._size_tnsr

            if self.use_half_precision:
                outputs = outputs.float()

            outputs = outputs.cpu()

        # Go through batches
        result_list = []

        if self.n_processes > 1:
            with PoolExecutor(self.n_processes) as ex:
                futures = []
                for i, output in enumerate(outputs):
                    # Normalized
                    preds = output[output[..., 4] > self.conf_threshold]
                    fut = ex.submit(
                        self.process_prediction, preds, preproc_data[i], self.nms, self.preproc_ops
                    )
                    futures.append(fut)

                for fut in futures:
                    preds = fut.result()
                    # Tuple of three components
                    result_list.append(preds)
        else:
            for i, output in enumerate(outputs):
                preds = output[output[..., 4] > self.conf_threshold]
                # print(f'Received {preds.shape} predictions')

                preds = self.process_prediction(preds, preproc_data[i], self.nms, self.preproc_ops)
                result_list.append(preds)

        return result_list

    @staticmethod
    def process_prediction(preds, preproc_data, nms, preproc_ops):
        if preds.shape[0] == 0:
            return {"bboxes": np.array([]), "classes": np.array([]), "scores": np.array([])}

        keep = nms.exec(preds)

        preds = preds[keep]  # .cpu()
        preds = {"bboxes": preds[:, :4], "classes": preds[:, 5].long(), "scores": preds[:, 4]}

        for op in reversed(preproc_ops):
            preds = op.inverse_transform(preds=preds, data=preproc_data)

        return preds
