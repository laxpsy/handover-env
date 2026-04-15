import math
import random

from entities.base_station import BaseStation
from entities.handover import UEHandoverState
from entities.network import Network
from entities.rendering_entities import Coordinates
from entities.ue import UE, UEMovementType
from simulation.config import SimulationConfig

LINEAR_MOBILITY_CONFIG = SimulationConfig(
    SCREEN_WIDTH=1200,
    SCREEN_HEIGHT=700,
    STEP=100,
    PATH_LOSS_EXPONENT=3.0,
    PATH_LOSS_REFERENCE_DISTANCE=1,
    HYSTERISIS_MARGIN=3.0,
    TIME_TO_TRIGGER=4,
    PING_PONG_WINDOW=4,
    RLF_FAILURE_THRESHOLD=-97,
    MAX_HISTORY=12,
    EARLY_HANDOVER_WINDOW=1200,
    MIN_HISTORY_LENGTH=3,
    DEFAULT_VELOCITY=1.5,
    DEFAULT_TX_POWER=53.0,
    DEFAULT_FREQUENCY=24.25e9,
)


REALISTIC_MOBILITY_CONFIG = SimulationConfig(
    SCREEN_WIDTH=1200,
    SCREEN_HEIGHT=700,
    STEP=100,
    PATH_LOSS_EXPONENT=3.2,
    PATH_LOSS_REFERENCE_DISTANCE=1,
    HYSTERISIS_MARGIN=4.0,
    TIME_TO_TRIGGER=5,
    PING_PONG_WINDOW=5,
    RLF_FAILURE_THRESHOLD=-97,
    MAX_HISTORY=14,
    EARLY_HANDOVER_WINDOW=1500,
    MIN_HISTORY_LENGTH=4,
    DEFAULT_VELOCITY=1.8,
    DEFAULT_TX_POWER=53.0,
    DEFAULT_FREQUENCY=24.25e9,
)


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (count - 1)
    return [start + step * idx for idx in range(count)]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _default_handover_state() -> UEHandoverState:
    return UEHandoverState(
        target_base_station=-1,
        ttt_timer=0.0,
        ttt_running=False,
        step_count_since_last_handover=0,
        handover_this_step=False,
    )


def _nearest_bs_id(
    ue_position: Coordinates,
    base_stations: list[BaseStation],
) -> int | None:
    if not base_stations:
        return None

    nearest_id = base_stations[0].id
    nearest_distance_sq = float("inf")

    for bs in base_stations:
        dx = ue_position.x - bs.coordinates.x
        dy = ue_position.y - bs.coordinates.y
        distance_sq = dx * dx + dy * dy
        if distance_sq < nearest_distance_sq:
            nearest_distance_sq = distance_sq
            nearest_id = bs.id

    return nearest_id


def _validate_counts(num_ues: int, num_bs: int) -> None:
    if not 12 <= num_ues <= 14:
        raise ValueError("num_ues must be between 12 and 14")
    if not 8 <= num_bs <= 10:
        raise ValueError("num_bs must be between 8 and 10")


def _create_linear_base_stations(
    num_bs: int,
    config: SimulationConfig,
) -> list[BaseStation]:
    margin_x = 120.0
    top_y = config.SCREEN_HEIGHT * 0.28
    bottom_y = config.SCREEN_HEIGHT * 0.72

    top_count = (num_bs + 1) // 2
    bottom_count = num_bs - top_count

    top_xs = _linspace(margin_x, config.SCREEN_WIDTH - margin_x, top_count)
    bottom_xs = _linspace(
        margin_x,
        config.SCREEN_WIDTH - margin_x,
        bottom_count,
    )

    base_stations: list[BaseStation] = []
    bs_id = 0

    for x in top_xs:
        base_stations.append(
            BaseStation(
                id=bs_id,
                coordinates=Coordinates(x=x, y=top_y),
                tx_power=config.DEFAULT_TX_POWER,
                transmission_frequency=config.DEFAULT_FREQUENCY,
            )
        )
        bs_id += 1

    for x in bottom_xs:
        base_stations.append(
            BaseStation(
                id=bs_id,
                coordinates=Coordinates(x=x, y=bottom_y),
                tx_power=config.DEFAULT_TX_POWER,
                transmission_frequency=config.DEFAULT_FREQUENCY,
            )
        )
        bs_id += 1

    return base_stations


def _realistic_bs_power(bs_index: int, num_bs: int) -> float:
    if num_bs == 9:
        power_map = [43.0, 53.0, 43.0, 53.0, 56.0, 53.0, 43.0, 53.0, 43.0]
        return power_map[bs_index]

    fallback_power_pattern = [43.0, 53.0, 56.0, 53.0]
    return fallback_power_pattern[bs_index % len(fallback_power_pattern)]


def _create_realistic_base_stations(
    num_bs: int,
    config: SimulationConfig,
    rng: random.Random,
) -> list[BaseStation]:
    margin_x = 120.0
    margin_y = 90.0
    jitter = 25.0

    columns = 3 if num_bs <= 9 else 4
    rows = math.ceil(num_bs / columns)

    x_slots = _linspace(
        margin_x,
        config.SCREEN_WIDTH - margin_x,
        columns,
    )
    y_slots = _linspace(
        margin_y,
        config.SCREEN_HEIGHT - margin_y,
        rows,
    )

    base_stations: list[BaseStation] = []
    for idx in range(num_bs):
        row = idx // columns
        col = idx % columns

        x = _clamp(
            x_slots[col] + rng.uniform(-jitter, jitter),
            margin_x / 2,
            config.SCREEN_WIDTH - margin_x / 2,
        )
        y = _clamp(
            y_slots[row] + rng.uniform(-jitter, jitter),
            margin_y / 2,
            config.SCREEN_HEIGHT - margin_y / 2,
        )

        base_stations.append(
            BaseStation(
                id=idx,
                coordinates=Coordinates(x=x, y=y),
                tx_power=_realistic_bs_power(idx, num_bs),
                transmission_frequency=config.DEFAULT_FREQUENCY,
            )
        )

    return base_stations


def create_network_linear(
    num_ues: int = 12,
    num_bs: int = 8,
    config: SimulationConfig = LINEAR_MOBILITY_CONFIG,
    seed: int | None = 7,
) -> Network:
    _validate_counts(num_ues, num_bs)

    rng = random.Random(seed)
    base_stations = _create_linear_base_stations(num_bs, config)

    lane_count = 4
    lanes = _linspace(
        config.SCREEN_HEIGHT * 0.2,
        config.SCREEN_HEIGHT * 0.8,
        lane_count,
    )
    x_slots = _linspace(80.0, config.SCREEN_WIDTH - 80.0, num_ues)

    ues: list[UE] = []
    for idx in range(num_ues):
        speed = 0.5 + (2.5 * idx / max(1, num_ues - 1))
        heading = 0.0 if idx % 2 == 0 else math.pi
        heading += ((idx % 5) - 2) * 0.08

        velocity_x = speed * math.cos(heading)
        velocity_y = speed * math.sin(heading)

        x = _clamp(
            x_slots[idx] + rng.uniform(-25.0, 25.0),
            40.0,
            config.SCREEN_WIDTH - 40.0,
        )
        y = _clamp(
            lanes[idx % lane_count] + rng.uniform(-18.0, 18.0),
            40.0,
            config.SCREEN_HEIGHT - 40.0,
        )

        ue_position = Coordinates(x=x, y=y)
        serving_bs = _nearest_bs_id(ue_position, base_stations)

        ues.append(
            UE(
                id=idx,
                coordinates=ue_position,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                serving_bs=serving_bs,
                rsrp={},
                movement_type=UEMovementType.Linear,
                handover_state=_default_handover_state(),
                handover_history=[],
                total_handovers=0,
            )
        )

    return Network(ues=ues, base_stations=base_stations)


def create_network_realistic(
    num_ues: int = 14,
    num_bs: int = 9,
    config: SimulationConfig = REALISTIC_MOBILITY_CONFIG,
    seed: int | None = 21,
) -> Network:
    _validate_counts(num_ues, num_bs)

    rng = random.Random(seed)
    base_stations = _create_realistic_base_stations(num_bs, config, rng)

    columns = math.ceil(math.sqrt(num_ues))
    rows = math.ceil(num_ues / columns)
    x_slots = _linspace(90.0, config.SCREEN_WIDTH - 90.0, columns)
    y_slots = _linspace(70.0, config.SCREEN_HEIGHT - 70.0, rows)

    random_count = max(1, num_ues // 2)
    movement_types = [UEMovementType.Random] * random_count + [
        UEMovementType.Linear
    ] * (num_ues - random_count)
    rng.shuffle(movement_types)

    ues: list[UE] = []
    for idx in range(num_ues):
        row = idx // columns
        col = idx % columns

        x = _clamp(
            x_slots[col] + rng.uniform(-35.0, 35.0),
            40.0,
            config.SCREEN_WIDTH - 40.0,
        )
        y = _clamp(
            y_slots[row] + rng.uniform(-30.0, 30.0),
            40.0,
            config.SCREEN_HEIGHT - 40.0,
        )

        profile = idx % 3
        if profile == 0:
            speed = rng.uniform(0.5, 1.2)
        elif profile == 1:
            speed = rng.uniform(1.3, 2.3)
        else:
            speed = rng.uniform(2.4, 4.0)

        heading = rng.uniform(0.0, 2.0 * math.pi)
        velocity_x = speed * math.cos(heading)
        velocity_y = speed * math.sin(heading)

        ue_position = Coordinates(x=x, y=y)
        serving_bs = _nearest_bs_id(ue_position, base_stations)

        ues.append(
            UE(
                id=idx,
                coordinates=ue_position,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                serving_bs=serving_bs,
                rsrp={},
                movement_type=movement_types[idx],
                handover_state=_default_handover_state(),
                handover_history=[],
                total_handovers=0,
            )
        )

    return Network(ues=ues, base_stations=base_stations)
