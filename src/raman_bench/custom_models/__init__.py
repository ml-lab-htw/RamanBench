"""Compatibility shim: ``raman_bench.custom_models`` → ``raman_bench.models.custom``.

Old import path preserved for backwards compatibility. New code should use
``raman_bench.models.custom`` directly.
"""

from raman_bench.models.custom import *  # noqa: F401, F403
from raman_bench.models.custom import (  # noqa: F401
    CUSTOM_MODELS,
    FlexiblePipelinePLS,
    PreprocessingPLS,
)
