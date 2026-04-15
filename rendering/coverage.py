import math
from typing import TYPE_CHECKING

from raylib import rl

from rendering.shapes import draw_circle_fill, draw_circle_outline

if TYPE_CHECKING:
    from entities.base_station import BaseStation
    from simulation.config import SimulationConfig


def compute_reference_path_loss(config: "SimulationConfig") -> float:
    wavelength = config.SPEED_OF_LIGHT / config.DEFAULT_FREQUENCY
    return 20 * math.log10(
        4 * math.pi * config.PATH_LOSS_REFERENCE_DISTANCE / wavelength
    )


def compute_coverage_radius(bs: "BaseStation", config: "SimulationConfig") -> float:
    target_rsrp = config.RLF_FAILURE_THRESHOLD
    tx_power = bs.tx_power

    pl_reference = compute_reference_path_loss(config)
    max_path_loss = tx_power - target_rsrp

    if max_path_loss <= pl_reference:
        return config.PATH_LOSS_REFERENCE_DISTANCE

    distance_ratio = 10 ** (
        (max_path_loss - pl_reference) / (10 * config.PATH_LOSS_EXPONENT)
    )
    radius = config.PATH_LOSS_REFERENCE_DISTANCE * distance_ratio

    return radius


def draw_coverage_region(
    bs: "BaseStation",
    radius: float,
    coverage_color,
    alpha: int,
    fill: bool = True,
) -> None:
    color_with_alpha = coverage_color
    if fill:
        draw_circle_fill(bs.coordinates.x, bs.coordinates.y, radius, color_with_alpha)
    else:
        draw_circle_outline(
            bs.coordinates.x, bs.coordinates.y, radius, color_with_alpha
        )


def draw_coverage_region_with_edge(
    bs: "BaseStation",
    radius: float,
    coverage_color,
    edge_color,
    alpha: int,
) -> None:
    color_with_alpha = coverage_color
    if alpha > 0:
        draw_circle_fill(bs.coordinates.x, bs.coordinates.y, radius, color_with_alpha)
    draw_circle_outline(bs.coordinates.x, bs.coordinates.y, radius, edge_color)
