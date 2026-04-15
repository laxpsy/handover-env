from typing import TYPE_CHECKING

from rendering.shapes import draw_rectangle_rounded, draw_text

if TYPE_CHECKING:
    from simulation.config import SimulationConfig
    from simulation.statistics import SimulationStatistics


PANEL_WIDTH = 200
PANEL_HEIGHT = 150
PANEL_PADDING = 15
LINE_HEIGHT = 20


def draw_statistics_overlay(
    statistics: "SimulationStatistics",
    step_count: int,
    x: float,
    y: float,
    font_size: int,
    text_color,
    bg_alpha: int = 200,
) -> None:
    panel_x = x
    panel_y = y

    bg_color = (240, 240, 240, bg_alpha)
    draw_rectangle_rounded(
        int(panel_x), int(panel_y), PANEL_WIDTH, PANEL_HEIGHT, 0.1, bg_color
    )

    current_y = panel_y + PANEL_PADDING

    draw_text(
        "STATISTICS",
        panel_x + PANEL_PADDING,
        current_y,
        font_size,
        text_color,
    )
    current_y += LINE_HEIGHT + 5

    draw_text(
        f"Step: {step_count}",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 2,
        text_color,
    )
    current_y += LINE_HEIGHT

    draw_text(
        f"Early: {statistics.early_handover_count}",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 2,
        text_color,
    )
    current_y += LINE_HEIGHT

    draw_text(
        f"Late: {statistics.late_handover_count}",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 2,
        text_color,
    )
    current_y += LINE_HEIGHT

    draw_text(
        f"PingPong: {statistics.ping_pong_handover_count}",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 2,
        text_color,
    )
    current_y += LINE_HEIGHT

    draw_text(
        f"Success: {statistics.successful_handover_count}",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 2,
        text_color,
    )


def draw_config_info(
    config: "SimulationConfig",
    x: float,
    y: float,
    font_size: int,
    text_color,
    bg_alpha: int = 200,
) -> None:
    panel_x = x
    panel_y = y

    panel_width = 220
    panel_height = 110

    bg_color = (240, 240, 240, bg_alpha)
    draw_rectangle_rounded(
        int(panel_x), int(panel_y), panel_width, panel_height, 0.1, bg_color
    )

    current_y = panel_y + PANEL_PADDING

    original_hyst = float(config.ORIGINAL_HYSTERISIS_MARGIN)
    original_ttt = float(config.ORIGINAL_TIME_TO_TRIGGER)

    if config.HYSTERISIS_MARGIN > original_hyst:
        hyst_color = (0, 180, 0, 255)
    elif config.HYSTERISIS_MARGIN < original_hyst:
        hyst_color = (200, 0, 0, 255)
    else:
        hyst_color = text_color

    if config.TIME_TO_TRIGGER > original_ttt:
        ttt_color = (0, 180, 0, 255)
    elif config.TIME_TO_TRIGGER < original_ttt:
        ttt_color = (200, 0, 0, 255)
    else:
        ttt_color = text_color

    draw_text(
        "CONFIG",
        panel_x + PANEL_PADDING,
        current_y,
        font_size,
        text_color,
    )
    current_y += LINE_HEIGHT + 5

    draw_text(
        f"HYST: {config.HYSTERISIS_MARGIN} (orig {original_hyst})",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 4,
        hyst_color,
    )
    current_y += LINE_HEIGHT - 2

    draw_text(
        f"TTT: {config.TIME_TO_TRIGGER} (orig {original_ttt})",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 4,
        ttt_color,
    )
    current_y += LINE_HEIGHT - 2

    draw_text(
        f"RLF: {config.RLF_FAILURE_THRESHOLD} dBm",
        panel_x + PANEL_PADDING,
        current_y,
        font_size - 4,
        text_color,
    )
