from dataclasses import dataclass

from raylib import colors as rc


@dataclass
class ColorPalette:
    bg: "rc.Color"
    bs: "rc.Color"
    ue: "rc.Color"
    coverage: "rc.Color"
    connection: "rc.Color"
    rlf_pulse: "rc.Color"
    text: "rc.Color"
    success: "rc.Color"
    early: "rc.Color"
    late: "rc.Color"
    ping_pong: "rc.Color"


RAYLIB_COLORS = {
    "LIGHTGRAY": rc.LIGHTGRAY,
    "GRAY": rc.GRAY,
    "DARKGRAY": rc.DARKGRAY,
    "YELLOW": rc.YELLOW,
    "GOLD": rc.GOLD,
    "ORANGE": rc.ORANGE,
    "PINK": rc.PINK,
    "RED": rc.RED,
    "MAROON": rc.MAROON,
    "GREEN": rc.GREEN,
    "LIME": rc.LIME,
    "DARKGREEN": rc.DARKGREEN,
    "SKYBLUE": rc.SKYBLUE,
    "BLUE": rc.BLUE,
    "DARKBLUE": rc.DARKBLUE,
    "PURPLE": rc.PURPLE,
    "VIOLET": rc.VIOLET,
    "DARKPURPLE": rc.DARKPURPLE,
    "BEIGE": rc.BEIGE,
    "BROWN": rc.BROWN,
    "DARKBROWN": rc.DARKBROWN,
    "MAGENTA": rc.MAGENTA,
    "RAYWHITE": rc.RAYWHITE,
    "WHITE": rc.WHITE,
    "BLACK": rc.BLACK,
    "BLANK": rc.BLANK,
    "MAGENTA": rc.MAGENTA,
    "RAYWHITE": rc.RAYWHITE,
}
