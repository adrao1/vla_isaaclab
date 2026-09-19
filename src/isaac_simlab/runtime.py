"""Runtime helpers shared by scenario launchers."""

import traceback

from isaaclab.envs import ManagerBasedRLEnv


class ScenarioEnv(ManagerBasedRLEnv):
    """Keep partial-construction errors visible instead of masking them during cleanup."""

    def __init__(self, cfg):
        self._construction_complete = False
        try:
            super().__init__(cfg)
        except BaseException:
            traceback.print_exc()
            raise
        self._construction_complete = True

    def __del__(self):
        if getattr(self, "_construction_complete", False):
            self.close()
