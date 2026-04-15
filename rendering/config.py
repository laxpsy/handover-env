from dataclasses import dataclass

from raylib import colors as rc

from rendering.colors import ColorPalette, RAYLIB_COLORS


@dataclass
class RendererConfig:
    fps: int = 60
    bg_color_name: str = "WHITE"
    bs_color_name: str = "BLACK"
    ue_color_name: str = "BLACK"
    coverage_color_name: str = "GRAY"
    connection_color_name: str = "BLACK"
    rlf_pulse_color_name: str = "RED"
    text_color_name: str = "BLACK"
    bs_radius: float = 20.0
    ue_radius: float = 8.0
    bs_icon_scale: float = 1.5
    ue_icon_scale: float = 2.0
    coverage_alpha: int = 50
    connection_thickness: float = 2.0
    font_size: int = 20
    use_smooth_positions: bool = True

    @property
    def bg_color(self):
        return RAYLIB_COLORS.get(self.bg_color_name, rc.WHITE)

    @property
    def bs_color(self):
        return RAYLIB_COLORS.get(self.bs_color_name, rc.BLACK)

    @property
    def ue_color(self):
        return RAYLIB_COLORS.get(self.ue_color_name, rc.BLACK)

    @property
    def coverage_color(self):
        return RAYLIB_COLORS.get(self.coverage_color_name, rc.GRAY)

    @property
    def connection_color(self):
        return RAYLIB_COLORS.get(self.connection_color_name, rc.BLACK)

    @property
    def rlf_pulse_color(self):
        return RAYLIB_COLORS.get(self.rlf_pulse_color_name, rc.RED)

    @property
    def text_color(self):
        return RAYLIB_COLORS.get(self.text_color_name, rc.BLACK)


@dataclass
class PaletteConfig:
    name: str = "default"
    colors: ColorPalette = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = DEFAULT_PALETTE


DEFAULT_PALETTE = ColorPalette(
    bg=rc.WHITE,
    bs=rc.BLACK,
    ue=rc.BLACK,
    coverage=rc.GRAY,
    connection=rc.BLACK,
    rlf_pulse=rc.RED,
    text=rc.BLACK,
    success=rc.GREEN,
    early=rc.YELLOW,
    late=rc.ORANGE,
    ping_pong=rc.MAGENTA,
)

NIGHT_MODE_PALETTE = ColorPalette(
    bg=rc.BLACK,
    bs=rc.DARKBLUE,
    ue=rc.LIGHTGRAY,
    coverage=rc.BLUE,
    connection=rc.GOLD,
    rlf_pulse=rc.RED,
    text=rc.WHITE,
    success=rc.GREEN,
    early=rc.YELLOW,
    late=rc.ORANGE,
    ping_pong=rc.MAGENTA,
)
