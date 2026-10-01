"""XSimLiftGrip with the YCB 006 mustard bottle in place of the sugar box.

Everything else (7-D grip controller task, force-capped Dex3 hand, X-Sim reward,
fixed reset) is inherited unchanged. Lives in the ycb_sugar_box package because
it reuses that package's scene, managers and MDP terms.

Orientation / height are derived from the mesh (rl/grasp_geometry.py), not tuned:
the mustard mesh is upright after a -90 deg turn about x (the same rotation the
scene preview uses), then the sugar box's table yaw is applied. The bottle rests
on its mesh bottom at SUPPORT_HEIGHT (0.62 m).

MustardGuidedXSimLiftGripEnvCfg adds pose guidance toward the nearest of several
saved, physics-validated grasps (assets/grasps/mustard_bottle.json). That
guidance is OUR addition, not X-Sim.
"""

from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .env_cfg import INITIAL_XY
from .guided_grasp_env_cfg import contact_reward
from .mdp import grasp_guidance
from .xsim_lift_env_cfg import XSimLiftGripEnvCfg

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MUSTARD_BOTTLE_USD = PROJECT_ROOT / "assets/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
MUSTARD_GRASP_LIBRARY = PROJECT_ROOT / "assets/grasps/mustard_bottle.json"

# upright_quat('x-90', SUGAR_BOX_TABLE_YAW_RAD) and 0.62 - min z of the rotated mesh.
MUSTARD_ORIENTATION_WXYZ = (
    0.6728436464587194,
    -0.6728436464587194,
    -0.21744292911045365,
    0.21744292911045365,
)
MUSTARD_REST_Z_M = 0.7156509950238988  # bottle is 0.1913 m tall

# Guidance weights are per-step (divided by step_dt below, like the X-Sim term).
GUIDE_REACH_WEIGHT = 1.0
GUIDE_ORIENTATION_WEIGHT = 0.5
GUIDE_CONTACT_WEIGHT = 1.0


@configclass
class MustardXSimLiftGripEnvCfg(XSimLiftGripEnvCfg):
    """Mustard bottle, upright, same task/reward/controller as XSimLiftGrip."""

    task_instruction: str = (
        "Grasp the YCB 006 mustard bottle with the left Dex3 hand and lift it 5 cm."
    )

    def __post_init__(self):
        super().__post_init__()
        if not MUSTARD_BOTTLE_USD.is_file():
            raise FileNotFoundError(f"Missing cached YCB asset: {MUSTARD_BOTTLE_USD}")
        self.scene.object.spawn.usd_path = str(MUSTARD_BOTTLE_USD)
        self.scene.object.init_state.pos = (INITIAL_XY[0], INITIAL_XY[1], MUSTARD_REST_Z_M)
        self.scene.object.init_state.rot = MUSTARD_ORIENTATION_WXYZ


@configclass
class MustardGuidedXSimLiftGripEnvCfg(MustardXSimLiftGripEnvCfg):
    """Mustard XSimLiftGrip plus pose guidance toward saved validated grasps."""

    task_instruction: str = (
        "Grasp the YCB 006 mustard bottle with the left Dex3 hand, guided toward "
        "saved validated grasps, and lift it 5 cm."
    )

    def __post_init__(self):
        super().__post_init__()
        if not MUSTARD_GRASP_LIBRARY.is_file():
            raise FileNotFoundError(
                f"Missing grasp library: {MUSTARD_GRASP_LIBRARY} (run rl/build_grasp_library.py)"
            )
        per_step = 1.0 / (self.sim.dt * self.decimation)
        params = {"library_path": str(MUSTARD_GRASP_LIBRARY)}
        self.rewards.guide_reach = RewTerm(
            func=grasp_guidance.guidance_reach, weight=GUIDE_REACH_WEIGHT * per_step, params=dict(params)
        )
        self.rewards.guide_orientation = RewTerm(
            func=grasp_guidance.guidance_orientation,
            weight=GUIDE_ORIENTATION_WEIGHT * per_step,
            params=dict(params),
        )
        self.rewards.guide_contact = RewTerm(func=contact_reward, weight=GUIDE_CONTACT_WEIGHT * per_step)
