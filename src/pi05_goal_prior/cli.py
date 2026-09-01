"""Entry points that import this package before handing over to lerobot.

`python -m lerobot.scripts.lerobot_train --policy.type=pi05_goal_prior` fails on its own:
lerobot has no reason to import a third-party package, so the config subclass is never
registered and draccus does not know the name. These wrappers import it first and then call
lerobot's own main, so every flag behaves exactly as documented upstream.
"""
import sys

import pi05_goal_prior  # noqa: F401  -- registers the policy type


def train() -> None:
    from lerobot.scripts.lerobot_train import main
    main()


def evaluate() -> None:
    from lerobot.scripts.lerobot_eval import main
    main()


if __name__ == "__main__":
    (evaluate if "--eval" in sys.argv else train)()
