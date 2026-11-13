
from pathlib import Path
import yaml
from dataclasses import dataclass
from environment_generation.visual_nuisances.appearence import Appearance, apply_appearance, sample_appearance
from .seed import SeedManager
from sim.world import World
from environment_generation.visual_nuisances.lighting import Lighting, sample_lighting
from environment_generation.visual_nuisances.distractor import apply_distractors, sample_distractors, Distractor
@dataclass(frozen=True)
class Scenario:
    root_seed: int
    appearance: Appearance
    appearance_enabled: bool
    # None means the renderer's own default lighting (the true baseline).
    lighting: Lighting | None
    lighting_enabled: bool
    distractors: tuple[Distractor, ...] = ()
    distractors_enabled: bool = False

    @property
    def any_enabled(self) -> bool:
        return (
            self.appearance_enabled
            or self.lighting_enabled
            or self.distractors_enabled
        )

def load_scenario_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise TypeError("Scenario configuration must be a mapping")

    for key in ("baseline", "nuisances"):
        if key not in data:
            raise ValueError(f"Missing configuration section: {key}")

    return data



def sample_scenario(
    config: dict,
    root_seed: int | None,
) -> Scenario:

    appearance_config = config["nuisances"]["appearance"]
    enabled = appearance_config["enabled"]

    if type(enabled) is not bool:
        raise TypeError("appearance.enabled must be a boolean")

    seed_manager = SeedManager(root_seed)
    resolved_seed = int(seed_manager.root_seed)

    if enabled:
        appearance = sample_appearance(
            appearance_config,
            seed_manager.rng("appearance")
        )

    else:
        baseline = config["baseline"]["appearance"]

        appearance = Appearance(
            block_color=tuple(baseline["block_color"]),
            floor_color=tuple(baseline["floor_color"]),
        )
    lighting_config = config["nuisances"]["lighting"]
    lighting_enabled = lighting_config["enabled"]

    if type(lighting_enabled) is not bool:
        raise TypeError("lighting.enabled must be a boolean")

    if lighting_enabled:
        lighting = sample_lighting(
        lighting_config,
        seed_manager.rng("lighting"),
    )
    else:
        lighting = None

    distractor_config = config["nuisances"]["distractors"]
    distractors_enabled = distractor_config["enabled"]

    if type(distractors_enabled) is not bool:
        raise TypeError("distractors.enabled must be a boolean")

    if distractors_enabled:
        distractors = sample_distractors(
            distractor_config,
            seed_manager.rng("distractors"),
        )
    else:
        distractors = ()

    return Scenario(
        root_seed=resolved_seed,
        appearance=appearance,
        appearance_enabled=enabled,
        lighting=lighting,
        lighting_enabled=lighting_enabled,
        distractors=distractors,
        distractors_enabled=distractors_enabled,
    )
def apply_scenario(world: World, scenario: Scenario | None) -> None:
    """Make the world's visuals exactly match the scenario.

    Always starts from the world's original visuals, so a disabled nuisance
    can never keep a previously applied value. Lighting is render-time state
    and is handled by the camera, not the world.
    """
    world.restore_default_visuals()
    if scenario is None:
        return
    if scenario.appearance_enabled:
        apply_appearance(world, scenario.appearance)
    if scenario.distractors_enabled:
        apply_distractors(world, scenario.distractors)




    

    
  