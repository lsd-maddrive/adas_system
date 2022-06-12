import json
import importlib
from torch import nn


def get_model_config(path_to_config: str):
    with open(path_to_config, 'rb') as f:
        config_data = f.read().decode('utf-8')
    model_data = json.loads(config_data)
    return model_data


def get_model_and_img_size(
    path_to_config: str = '',
    config_data: str = ''
) -> tuple[nn.Module, int]:
    assert(path_to_config or config_data), f'{__name__} Unable to get model: empty args.'
    if path_to_config:
        model_data = get_model_config(path_to_config)
    else:
        # TODO: fix double convert
        model_data = json.loads(json.loads(config_data))

    model = getattr(
        importlib.import_module('torchvision.models'),
        f'{model_data["base"]}')(pretrained=True)
    if 'efficientnet' in model_data["base"]:
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=model_data['output_len'],
            bias=model_data['bias'])
    elif 'resnet' in model_data["base"]:
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=model_data['output_len'],
            bias=model_data['bias'])
    img_size = model_data['input_image_size']
    return model, img_size
