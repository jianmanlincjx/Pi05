"""Two-stage goal-pose prior on pi05, as a LeRobot policy plugin.

Importing this package is what registers `--policy.type=pi05_goal_prior`: the config class
carries `@PreTrainedConfig.register_subclass`, and lerobot resolves a policy name to its
class by convention from the registered config's module path. lerobot never imports this
package on its own, so run training and evaluation through the `pi05gp-train` /
`pi05gp-eval` entry points, or `import pi05_goal_prior` before calling lerobot's own.
"""

from .configuration_pi05_goal_prior import PI05GoalPriorConfig as PI05GoalPriorConfig

__all__ = ["PI05GoalPriorConfig"]
