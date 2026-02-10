import importlib


def load_adapter(model_name: str):
    """
    Resolution order:
      1) built-in framework adapter: framework.adapters.<model_name>
         expecting class Adapter
      2) model-local adapter: models/<model_name>/adapter.py
         expecting Adapter or <MODEL>Adapter
    """
    # 1) framework adapters first
    try:
        module = importlib.import_module(f"framework.adapters.{model_name}")
        if hasattr(module, "Adapter"):
            return module.Adapter()
    except ModuleNotFoundError:
        pass

    # 2) fallback to model-local adapter (optional)
    module = importlib.import_module(f"models.{model_name}.adapter")

    if hasattr(module, "Adapter"):
        return module.Adapter()

    candidate = f"{model_name.upper()}Adapter"
    if hasattr(module, candidate):
        return getattr(module, candidate)()

    candidate2 = f"{model_name.capitalize()}Adapter"
    if hasattr(module, candidate2):
        return getattr(module, candidate2)()

    raise AttributeError(
        f"No adapter found for model '{model_name}'. "
        f"Expected framework/adapters/{model_name}.py with Adapter class "
        f"or models/{model_name}/adapter.py."
    )