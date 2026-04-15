from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from raylib import rl

from rendering.coverage import (compute_coverage_radius,
                                draw_coverage_region_with_edge)
from rendering.entities import (draw_base_station_icon, draw_connection_line,
                                draw_ue_icon, get_rlf_pulse_intensity,
                                unload_icon_textures)
from rendering.statistics import draw_config_info, draw_statistics_overlay

if TYPE_CHECKING:
    from rendering.config import RendererConfig
    from simulation.simulator import Simulation


rl.SetConfigFlags(rl.FLAG_WINDOW_RESIZABLE)


@dataclass
class PreviousState:
    ue_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    step_count: int = 0


class Renderer:
    def __init__(
        self,
        simulation: "Simulation",
        config: Optional["RendererConfig"] = None,
        window_title: str = "Handover Simulation",
    ) -> None:
        from rendering.config import RendererConfig

        self.simulation = simulation
        self.config = config or RendererConfig()
        self.window_title = window_title
        self._window_open = False
        self._initialized = False
        self._previous_state = PreviousState()
        self._pinned_ue_id: int | None = None
        self._pinned_bs_id: int | None = None

        self._init_window()

    def _init_window(self) -> None:
        screen_width = self.simulation.config.SCREEN_WIDTH
        screen_height = self.simulation.config.SCREEN_HEIGHT

        rl.InitWindow(screen_width, screen_height, self.window_title.encode())
        rl.SetTargetFPS(self.config.fps)

        self._window_open = True
        self._initialized = True

    def _store_previous_positions(self) -> None:
        for ue in self.simulation.state_space.ues:
            self._previous_state.ue_positions[ue.id] = (
                ue.coordinates.x,
                ue.coordinates.y,
            )
        self._previous_state.step_count = self.simulation.get_step_count()

    def _get_interpolated_position(
        self,
        ue_id: int,
        current_x: float,
        current_y: float,
    ) -> tuple[float, float]:
        if not self.config.use_smooth_positions:
            return current_x, current_y

        prev = self._previous_state.ue_positions.get(ue_id)
        if (
            prev is None
            or self._previous_state.step_count
            == self.simulation.get_step_count()
        ):
            return current_x, current_y

        prev_x, prev_y = prev
        alpha = 0.5
        interp_x = prev_x + alpha * (current_x - prev_x)
        interp_y = prev_y + alpha * (current_y - prev_y)

        return interp_x, interp_y

    def step(self) -> bool:
        if not self._window_open:
            return False

        if rl.WindowShouldClose():
            self._window_open = False
            return False

        self._handle_clicks()

        rl.BeginDrawing()
        rl.ClearBackground(self.config.bg_color)

        self._draw_background()
        self._draw_coverage_regions()
        self._draw_connections()
        self._draw_entities()
        self._draw_pinned_panels()
        self._draw_statistics()
        self._draw_hover_tooltip()

        rl.EndDrawing()

        self._store_previous_positions()

        return self._window_open

    def _get_mouse_position(self) -> tuple[float, float]:
        return float(rl.GetMouseX()), float(rl.GetMouseY())

    def _is_point_in_rect(
        self,
        px: float,
        py: float,
        rx: float,
        ry: float,
        rw: float,
        rh: float,
    ) -> bool:
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    def _is_point_in_circle(
        self,
        px: float,
        py: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> bool:
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy <= radius * radius

    def _find_hovered_ue(self, mouse_x: float, mouse_y: float):
        from entities.rendering_entities import Coordinates

        ue_radius = self.config.ue_radius * self.config.ue_icon_scale
        for ue in self.simulation.state_space.ues:
            interp_x, interp_y = self._get_interpolated_position(
                ue.id,
                ue.coordinates.x,
                ue.coordinates.y,
            )
            ue_pos = Coordinates(x=interp_x, y=interp_y)
            if self._is_point_in_circle(
                mouse_x,
                mouse_y,
                ue_pos.x,
                ue_pos.y,
                ue_radius,
            ):
                return ue
        return None

    def _find_ue_by_id(self, ue_id: int):
        for ue in self.simulation.state_space.ues:
            if ue.id == ue_id:
                return ue
        return None

    def _find_bs_by_id(self, bs_id: int):
        for bs in self.simulation.state_space.base_stations:
            if bs.id == bs_id:
                return bs
        return None

    def _get_ue_panel_geometry(self) -> tuple[int, int, int, int]:
        screen_width = self.simulation.config.SCREEN_WIDTH
        screen_height = self.simulation.config.SCREEN_HEIGHT
        panel_width = int(min(560, screen_width - 40))
        panel_height = 110
        panel_x = int((screen_width - panel_width) / 2)
        panel_y = int(screen_height - panel_height - 10)
        return panel_x, panel_y, panel_width, panel_height

    def _get_bs_panel_geometry(
        self,
        ue_panel_geometry: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        screen_width = self.simulation.config.SCREEN_WIDTH
        screen_height = self.simulation.config.SCREEN_HEIGHT
        panel_width = int(min(560, screen_width - 40))
        panel_height = 110
        panel_x = int((screen_width - panel_width) / 2)
        panel_y = int(screen_height - panel_height - 10)

        if ue_panel_geometry is not None:
            _, ue_panel_y, _, _ = ue_panel_geometry
            panel_y = int(ue_panel_y - panel_height - 5)

        return panel_x, panel_y, panel_width, panel_height

    def _handle_clicks(self) -> None:
        mouse_x, mouse_y = self._get_mouse_position()

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            hovered_ue = self._find_hovered_ue(mouse_x, mouse_y)
            if hovered_ue is not None:
                self._pinned_ue_id = hovered_ue.id
            else:
                hovered_bs = self._find_hovered_bs(mouse_x, mouse_y)
                if hovered_bs is not None:
                    self._pinned_bs_id = hovered_bs.id

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_RIGHT):
            ue_panel_clicked = False
            bs_panel_clicked = False

            if self._pinned_ue_id is not None:
                panel_x, panel_y, panel_width, panel_height = (
                    self._get_ue_panel_geometry()
                )
                if self._is_point_in_rect(
                    mouse_x,
                    mouse_y,
                    panel_x,
                    panel_y,
                    panel_width,
                    panel_height,
                ):
                    ue_panel_clicked = True

            if self._pinned_bs_id is not None:
                ue_geometry = None
                if self._pinned_ue_id is not None:
                    ue_geometry = self._get_ue_panel_geometry()
                panel_x, panel_y, panel_width, panel_height = (
                    self._get_bs_panel_geometry(ue_geometry)
                )
                if self._is_point_in_rect(
                    mouse_x,
                    mouse_y,
                    panel_x,
                    panel_y,
                    panel_width,
                    panel_height,
                ):
                    bs_panel_clicked = True

            if ue_panel_clicked:
                self._pinned_ue_id = None
            if bs_panel_clicked:
                self._pinned_bs_id = None

    def _find_hovered_bs(self, mouse_x: float, mouse_y: float):
        bs_radius = self.config.bs_radius * self.config.bs_icon_scale
        for bs in self.simulation.state_space.base_stations:
            if self._is_point_in_circle(
                mouse_x,
                mouse_y,
                bs.coordinates.x,
                bs.coordinates.y,
                bs_radius,
            ):
                return bs
        return None

    def _get_serving_bs(self, ue):
        if ue.serving_bs is None:
            return None
        for bs in self.simulation.state_space.base_stations:
            if bs.id == ue.serving_bs:
                return bs
        return None

    def _build_ue_tooltip_lines(self, ue) -> list[str]:
        lines = [f"UE{ue.id}"]
        serving_bs = self._get_serving_bs(ue)
        if serving_bs is None:
            lines.append("Connected BS: None")
            lines.append("RSRP: N/A")
            return lines

        lines.append(f"Connected BS: BS{serving_bs.id}")
        rsrp_value = ue.rsrp.get(serving_bs.id)
        if rsrp_value is None:
            lines.append("RSRP: N/A")
        else:
            lines.append(f"RSRP: {rsrp_value:.2f} dBm")
        return lines

    def _build_bs_tooltip_lines(self, bs) -> list[str]:
        connected_ues = [
            ue.id
            for ue in self.simulation.state_space.ues
            if ue.serving_bs == bs.id
        ]

        lines = [f"BS{bs.id}", f"Connected UEs: {len(connected_ues)}"]
        if connected_ues:
            ue_list = ", ".join(f"UE{ue_id}" for ue_id in connected_ues)
            lines.append(ue_list)
        else:
            lines.append("None")
        return lines

    def _draw_tooltip(
        self, lines: list[str], mouse_x: float, mouse_y: float
    ) -> None:
        if not lines:
            return

        padding = 8
        font_size = max(10, self.config.font_size - 6)
        line_height = font_size + 4

        max_line_width = 0
        for line in lines:
            line_width = rl.MeasureText(line.encode(), font_size)
            if line_width > max_line_width:
                max_line_width = line_width

        panel_width = max_line_width + padding * 2
        panel_height = line_height * len(lines) + padding * 2

        tooltip_x = int(mouse_x + 14)
        tooltip_y = int(mouse_y + 14)

        screen_width = self.simulation.config.SCREEN_WIDTH
        screen_height = self.simulation.config.SCREEN_HEIGHT

        if tooltip_x + panel_width > screen_width:
            tooltip_x = int(mouse_x - panel_width - 14)
        if tooltip_y + panel_height > screen_height:
            tooltip_y = int(mouse_y - panel_height - 14)

        bg_color = (240, 240, 240, 220)
        border_color = self.config.connection_color

        rl.DrawRectangle(
            tooltip_x, tooltip_y, panel_width, panel_height, bg_color
        )
        rl.DrawRectangleLines(
            tooltip_x,
            tooltip_y,
            panel_width,
            panel_height,
            border_color,
        )

        text_y = tooltip_y + padding
        for line in lines:
            rl.DrawText(
                line.encode(),
                tooltip_x + padding,
                text_y,
                font_size,
                self.config.text_color,
            )
            text_y += line_height

    def _draw_hover_tooltip(self) -> None:
        mouse_x, mouse_y = self._get_mouse_position()

        hovered_ue = self._find_hovered_ue(mouse_x, mouse_y)
        if hovered_ue is not None:
            self._draw_tooltip(
                self._build_ue_tooltip_lines(hovered_ue),
                mouse_x,
                mouse_y,
            )
            return

        hovered_bs = self._find_hovered_bs(mouse_x, mouse_y)
        if hovered_bs is not None:
            self._draw_tooltip(
                self._build_bs_tooltip_lines(hovered_bs),
                mouse_x,
                mouse_y,
            )

    def _build_pinned_ue_info_lines(self, ue) -> list[str]:
        serving_bs = self._get_serving_bs(ue)
        if serving_bs is None:
            serving_bs_text = "Connected BS: None"
            rsrp_text = "RSRP: N/A"
        else:
            serving_bs_text = f"Connected BS: BS{serving_bs.id}"
            rsrp_value = ue.rsrp.get(serving_bs.id)
            if rsrp_value is None:
                rsrp_text = "RSRP: N/A"
            else:
                rsrp_text = f"RSRP: {rsrp_value:.2f} dBm"

        movement_type = ue.movement_type.name
        speed = (
            ue.velocity_x * ue.velocity_x + ue.velocity_y * ue.velocity_y
        ) ** 0.5

        return [
            f"Pinned UE{ue.id}",
            f"{serving_bs_text}  |  {rsrp_text}",
            f"Movement: {movement_type}  |  Speed: {speed:.2f} px/step",
            f"Total HOs: {ue.total_handovers}  |  Since last HO: {
                ue.handover_state.step_count_since_last_handover} steps",
            "Right-click this panel to unpin",
        ]

    def _build_pinned_bs_info_lines(self, bs) -> list[str]:
        connected_ues = [
            ue.id
            for ue in self.simulation.state_space.ues
            if ue.serving_bs == bs.id
        ]
        frequency_ghz = bs.transmission_frequency / 1e9
        coverage_radius = compute_coverage_radius(bs, self.simulation.config)
        connected_text = (
            ", ".join(f"UE{ue_id}" for ue_id in connected_ues)
            if connected_ues
            else "None"
        )
        return [
            f"Pinned BS{bs.id}",
            f"TX: {bs.tx_power:.1f} dBm  |  Freq: {frequency_ghz:.2f} GHz",
            f"Coverage Radius: {coverage_radius:.0f}px  |  Connected UEs: {
                len(connected_ues)}",
            f"UE List: {connected_text}",
            "Right-click this panel to unpin",
        ]

    def _draw_info_panel(
        self,
        panel_x: int,
        panel_y: int,
        panel_width: int,
        panel_height: int,
        lines: list[str],
    ) -> None:
        bg_color = (240, 240, 240, 220)
        border_color = self.config.connection_color
        text_color = self.config.text_color

        rl.DrawRectangle(panel_x, panel_y, panel_width, panel_height, bg_color)
        rl.DrawRectangleLines(
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            border_color,
        )

        padding_x = 12
        padding_y = 10
        line_height = 18
        font_size = max(10, self.config.font_size - 7)

        text_y = panel_y + padding_y
        for line in lines:
            rl.DrawText(
                line.encode(),
                panel_x + padding_x,
                text_y,
                font_size,
                text_color,
            )
            text_y += line_height

    def _draw_pinned_panels(self) -> None:
        ue_panel_geometry = None

        if self._pinned_ue_id is not None:
            ue_panel_geometry = self._get_ue_panel_geometry()

        if self._pinned_bs_id is not None:
            pinned_bs = self._find_bs_by_id(self._pinned_bs_id)
            if pinned_bs is None:
                self._pinned_bs_id = None
            else:
                panel_x, panel_y, panel_width, panel_height = (
                    self._get_bs_panel_geometry(ue_panel_geometry)
                )
                self._draw_info_panel(
                    panel_x,
                    panel_y,
                    panel_width,
                    panel_height,
                    self._build_pinned_bs_info_lines(pinned_bs),
                )

        if self._pinned_ue_id is not None:
            pinned_ue = self._find_ue_by_id(self._pinned_ue_id)
            if pinned_ue is None:
                self._pinned_ue_id = None
            else:
                if ue_panel_geometry is None:
                    ue_panel_geometry = self._get_ue_panel_geometry()
                panel_x, panel_y, panel_width, panel_height = ue_panel_geometry
                self._draw_info_panel(
                    panel_x,
                    panel_y,
                    panel_width,
                    panel_height,
                    self._build_pinned_ue_info_lines(pinned_ue),
                )

    def _draw_background(self) -> None:
        pass

    def _draw_coverage_regions(self) -> None:
        coverage_color = self.config.coverage_color
        alpha = self.config.coverage_alpha

        coverage_color_alpha = (
            coverage_color[0],
            coverage_color[1],
            coverage_color[2],
            alpha,
        )

        for bs in self.simulation.state_space.base_stations:
            radius = compute_coverage_radius(bs, self.simulation.config)
            draw_coverage_region_with_edge(
                bs,
                radius,
                coverage_color_alpha,
                coverage_color,
                alpha,
            )

    def _draw_connections(self) -> None:
        from entities.rendering_entities import Coordinates

        normal_connection_color = self.config.connection_color
        thickness = self.config.connection_thickness
        step_count = self.simulation.get_step_count()
        rlf_threshold = self.simulation.config.RLF_FAILURE_THRESHOLD

        for ue in self.simulation.state_space.ues:
            if ue.serving_bs is None:
                continue

            serving_bs = None
            for bs in self.simulation.state_space.base_stations:
                if bs.id == ue.serving_bs:
                    serving_bs = bs
                    break

            if serving_bs is None:
                continue

            interp_x, interp_y = self._get_interpolated_position(
                ue.id, ue.coordinates.x, ue.coordinates.y
            )

            ue_pos = Coordinates(x=interp_x, y=interp_y)

            connection_color = normal_connection_color
            rsrp_value = ue.rsrp.get(ue.serving_bs)
            if rsrp_value is not None and rsrp_value < rlf_threshold:
                pulse = get_rlf_pulse_intensity(step_count)
                alpha = max(60, int(220 * pulse))
                connection_color = (255, 0, 0, alpha)
            elif ue.handover_state.handover_flash_steps > 0:
                max_flash_steps = 12.0
                flash_ratio = min(
                    1.0,
                    ue.handover_state.handover_flash_steps / max_flash_steps,
                )
                alpha = max(170, int(255 * flash_ratio))
                connection_color = (0, 255, 0, alpha)

            draw_connection_line(
                ue_pos,
                serving_bs,
                connection_color,
                thickness,
                dashed=True,
            )

    def _draw_entities(self) -> None:
        from entities.rendering_entities import Coordinates

        bs_radius = self.config.bs_radius
        bs_color = self.config.bs_color
        bs_scale = self.config.bs_icon_scale
        ue_radius = self.config.ue_radius
        ue_color = self.config.ue_color
        ue_scale = self.config.ue_icon_scale

        for bs in self.simulation.state_space.base_stations:
            draw_base_station_icon(
                bs,
                bs_radius,
                bs_color,
                show_id=True,
                scale=bs_scale,
            )

        for ue in self.simulation.state_space.ues:
            interp_x, interp_y = self._get_interpolated_position(
                ue.id, ue.coordinates.x, ue.coordinates.y
            )

            ue_pos = Coordinates(x=interp_x, y=interp_y)

            draw_ue_icon(
                ue_pos,
                ue_radius,
                ue_color,
                show_id=True,
                ue_id=ue.id,
                scale=ue_scale,
            )

    def _draw_statistics(self) -> None:
        screen_width = self.simulation.config.SCREEN_WIDTH
        stats_x = screen_width - 220
        stats_y = 20

        draw_statistics_overlay(
            self.simulation.statistics,
            self.simulation.get_step_count(),
            stats_x,
            stats_y,
            self.config.font_size,
            self.config.text_color,
        )

        config_x = stats_x
        config_y = stats_y + 190

        draw_config_info(
            self.simulation.config,
            config_x,
            config_y,
            self.config.font_size,
            self.config.text_color,
        )

    def is_window_open(self) -> bool:
        return self._window_open and not rl.WindowShouldClose()

    def close(self) -> None:
        if self._initialized:
            self._pinned_ue_id = None
            self._pinned_bs_id = None
            unload_icon_textures()
            rl.CloseWindow()
            self._window_open = False
            self._initialized = False

    def __del__(self) -> None:
        self.close()
