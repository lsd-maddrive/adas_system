def construct_model(d: dict, **default_kwargs: dict):
    import pydoc

    COMMON_PATH = "maddrive_adas.sign_det.models"

    assert isinstance(d, dict) and "type" in d
    kwargs = d.copy()
    model_type = kwargs.pop("type")

    if "Yolo4" in model_type:
        model_type = f"{COMMON_PATH}.yolo4.{model_type}"

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    constructor = pydoc.locate(model_type)
    if constructor is None:
        raise NotImplementedError(f"Model {model_type} not found")

    return constructor(**kwargs)
