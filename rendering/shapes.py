from raylib import ffi, rl


def _create_vector2(x: float, y: float):
    return ffi.new("Vector2 *", [x, y])[0]


def _create_rectangle(x: float, y: float, width: float, height: float):
    return ffi.new("Rectangle *", [x, y, width, height])[0]


def draw_circle_outline(center_x: float, center_y: float, radius: float, color) -> None:
    rl.DrawCircleLines(int(center_x), int(center_y), radius, color)


def draw_circle_fill(center_x: float, center_y: float, radius: float, color) -> None:
    rl.DrawCircle(int(center_x), int(center_y), int(radius), color)


def draw_circle_gradient(
    center_x: float, center_y: float, radius: float, color_inner, color_outer
) -> None:
    rl.DrawCircleGradient(
        int(center_x), int(center_y), radius, color_inner, color_outer
    )


def draw_line(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color,
    thickness: float = 1.0,
) -> None:
    rl.DrawLineEx(
        _create_vector2(start_x, start_y),
        _create_vector2(end_x, end_y),
        thickness,
        color,
    )


def draw_line_dashed(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color,
    dash_length: float = 5.0,
    gap_length: float = 3.0,
    thickness: float = 1.0,
) -> None:
    dx = end_x - start_x
    dy = end_y - start_y
    distance = (dx * dx + dy * dy) ** 0.5

    if distance == 0:
        return

    ux = dx / distance
    uy = dy / distance

    current_distance = 0.0
    is_dash = True
    current_x, current_y = start_x, start_y

    while current_distance < distance:
        segment_length = dash_length if is_dash else gap_length
        remaining_distance = distance - current_distance
        actual_length = min(segment_length, remaining_distance)

        next_x = current_x + ux * actual_length
        next_y = current_y + uy * actual_length

        if is_dash:
            draw_line(current_x, current_y, next_x, next_y, color, thickness)

        current_x, current_y = next_x, next_y
        current_distance += actual_length
        is_dash = not is_dash


def draw_text(text: str, pos_x: float, pos_y: float, font_size: int, color) -> None:
    rl.DrawText(text.encode(), int(pos_x), int(pos_y), font_size, color)


def draw_text_centered(
    text: str, center_x: float, center_y: float, font_size: int, color
) -> None:
    rl.DrawText(
        text.encode(),
        int(center_x - rl.MeasureText(text.encode(), font_size) / 2),
        int(center_y - font_size / 2),
        font_size,
        color,
    )


def draw_rectangle_outline(
    pos_x: float,
    pos_y: float,
    width: float,
    height: float,
    color,
    line_thickness: float = 1.0,
) -> None:
    rl.DrawRectangleLines(int(pos_x), int(pos_y), int(width), int(height), color)


def draw_rectangle_fill(
    pos_x: float, pos_y: float, width: float, height: float, color
) -> None:
    rl.DrawRectangle(int(pos_x), int(pos_y), int(width), int(height), color)


def draw_rectangle_rounded(
    pos_x: float, pos_y: float, width: float, height: float, roundness: float, color
) -> None:
    rect = _create_rectangle(pos_x, pos_y, width, height)
    rl.DrawRectangleRounded(
        rect,
        roundness,
        12,
        color,
    )
