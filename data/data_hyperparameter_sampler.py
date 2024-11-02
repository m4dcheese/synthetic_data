import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ForbiddenGreaterThanRelation
from config import make_dotdict_recursive


class DataHyperparameterSampler:
    def __init__(self, data_config):
        self.data_config = data_config

        self.config_space = CS.ConfigurationSpace()

        # hyperparameter: features (min, max) uniform constant or integer
        # hyperparameter: classes (min, max) uniform constant or integer
        # hyperparameter: samples (min, max) uniform constant or integer

        self.config_space.add_hyperparameters(
            [
                self._uniform_or_constant(
                    "features",
                    self.data_config.features.min,
                    self.data_config.features.max,
                ),
                self._uniform_or_constant(
                    "classes",
                    self.data_config.classes.min,
                    self.data_config.classes.max,
                ),
                self._uniform_or_constant(
                    "samples",
                    self.data_config.samples.min,
                    self.data_config.samples.max,
                ),
                # self._uniform_or_constant("t", 0, 1),
                # make t a float between 0 and 1
                CSH.UniformFloatHyperparameter(
                    name="t",
                    lower=0,
                    upper=1,
                ),
            ]
        )

    def _uniform_or_constant(self, name, lower, upper):
        hp = None
        if lower < upper:
            hp = CSH.UniformIntegerHyperparameter(
                name,
                lower=lower,
                upper=upper,
            )
        else:  # lower == upper:
            hp = CSH.Constant(name, lower)
        return hp

    def sample(self):
        sample_dict = dict(self.config_space.sample_configuration())
        return make_dotdict_recursive(sample_dict)
