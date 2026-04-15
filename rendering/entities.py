from pathlib import Path
from typing import TYPE_CHECKING

from raylib import rl

from rendering.shapes import (
    _create_vector2,
    draw_line_dashed,
    draw_text_centered,
)

if TYPE_CHECKING:
    from entities.base_station import BaseStation
    from entities.ue import UE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BS_ICON_PATH = str(PROJECT_ROOT / "bs.png")
UE_ICON_PATH = str(PROJECT_ROOT / "ue.png")
_TEXTURE_CACHE: dict[str, object] = {}


def _load_icon_texture(path: str):
    texture = _TEXTURE_CACHE.get(path)
    if texture is None:
        texture = rl.LoadTexture(path.encode())
        _TEXTURE_CACHE[path] = texture
    return texture


def unload_icon_textures() -> None:
    for texture in _TEXTURE_CACHE.values():
        rl.UnloadTexture(texture)
    _TEXTURE_CACHE.clear()


def _draw_textured_icon(
    path: str,
    center_x: float,
    center_y: float,
    target_size: float,
    tint=(255, 255, 255, 255),
) -> None:
    texture = _load_icon_texture(path)
    source_size = float(max(texture.width, texture.height))
    if source_size <= 0:
        return
    scale = target_size / source_size
    draw_x = center_x - (texture.width * scale) / 2.0
    draw_y = center_y - (texture.height * scale) / 2.0
    rl.DrawTextureEx(
        texture,
        _create_vector2(draw_x, draw_y),
        0.0,
        scale,
        tint,
    )


def draw_base_station_icon(
    bs: "BaseStation",
    radius: float,
    color,
    show_id: bool = True,
    scale: float = 1.0,
) -> None:
    icon_target_size = radius * 2.0 * scale
    _draw_textured_icon(
        BS_ICON_PATH,
        bs.coordinates.x,
        bs.coordinates.y,
        target_size=icon_target_size,
    )
    if show_id:
        draw_text_centered(
            f"BS{bs.id}",
            bs.coordinates.x,
            bs.coordinates.y + icon_target_size / 2.0 + 10.0,
            12,
            color,
        )


def draw_ue_icon(
    ue_pos,
    radius: float,
    color,
    show_id: bool = False,
    ue_id: int = 0,
    scale: float = 1.0,
) -> None:
    icon_target_size = radius * 2.0 * scale
    _draw_textured_icon(
        UE_ICON_PATH,
        ue_pos.x,
        ue_pos.y,
        target_size=icon_target_size,
    )
    if show_id:
        draw_text_centered(
            f"UE{ue_id}",
            ue_pos.x,
            ue_pos.y + icon_target_size / 2.0 + 10.0,
            10,
            color,
        )


def draw_connection_line(
    ue_pos,
    bs: "BaseStation",
    connection_color,
    thickness: float = 2.0,
    dashed: bool = True,
) -> None:
    if dashed:
        draw_line_dashed(
            ue_pos.x,
            ue_pos.y,
            bs.coordinates.x,
            bs.coordinates.y,
            connection_color,
            dash_length=5.0,
            gap_length=3.0,
            thickness=thickness,
        )
    else:
        rl.DrawLineEx(
            _create_vector2(ue_pos.x, ue_pos.y),
            _create_vector2(bs.coordinates.x, bs.coordinates.y),
            thickness,
            connection_color,
        )


def get_rlf_pulse_intensity(step_count: int, pulse_period: int = 30) -> float:
    import math

    phase = (step_count % pulse_period) / pulse_period
    return abs(math.sin(phase * math.pi * 2))
