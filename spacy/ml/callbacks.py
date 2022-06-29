from typing import Callable, TYPE_CHECKING
import warnings

from thinc.layers import with_nvtx_range, with_signpost_interval
from thinc.model import Model, wrap_model_recursive

from ..errors import Warnings
from ..util import registry

if TYPE_CHECKING:
    # This lets us add type hints for mypy etc. without causing circular imports
    from ..language import Language  # noqa: F401


@registry.callbacks("spacy.models_with_nvtx_range.v1")
def create_models_with_nvtx_range(
    forward_color: int = -1, backprop_color: int = -1
) -> Callable[["Language"], "Language"]:
    def models_with_nvtx_range(nlp):
        pipes = [
            pipe
            for _, pipe in nlp.components
            if hasattr(pipe, "is_trainable") and pipe.is_trainable
        ]

        # We need process all models jointly to avoid wrapping callbacks twice.
        models = Model(
            "wrap_with_nvtx_range",
            forward=lambda model, X, is_train: ...,
            layers=[pipe.model for pipe in pipes],
        )

        for node in models.walk():
            with_nvtx_range(
                node, forward_color=forward_color, backprop_color=backprop_color
            )

        return nlp

    return models_with_nvtx_range


@registry.callbacks("spacy.models_with_signpost_interval.v1")
def create_models_with_nvtx_range() -> Callable[["Language"], "Language"]:
    def models_with_nvtx_range(nlp):
        try:
            from os_signpost import SignPoster
        except ImportError:
            warnings.warn(Warnings.W121)
            return nlp

        log = SignPoster("ai.explosion.thinc", SignPoster.Category.DynamicTracing)

        pipes = [
            pipe
            for _, pipe in nlp.components
            if hasattr(pipe, "is_trainable") and pipe.is_trainable
        ]

        # We need process all models jointly to avoid wrapping callbacks twice.
        models = Model(
            "wrap_with_signpost",
            forward=lambda model, X, is_train: ...,
            layers=[pipe.model for pipe in pipes],
        )

        for node in models.walk():
            with_signpost_interval(node, log)

        return nlp

    return models_with_nvtx_range
