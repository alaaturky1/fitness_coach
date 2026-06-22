from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """
    Returns the angle ABC in degrees (0..180).
    Robust to scale; undefined when vectors have ~0 length (returns NaN).
    """

    bax_x = a.x - b.x
    bax_y = a.y - b.y
    bcx_x = c.x - b.x
    bcx_y = c.y - b.y

    dot = bax_x * bcx_x + bax_y * bcx_y
    norm1 = math.hypot(bax_x, bax_y)
    norm2 = math.hypot(bcx_x, bcx_y)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return float("nan")

    cosv = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cosv))


def vector_angle_to_vertical_degrees(p_from: Point, p_to: Point) -> float:
    """
    Angle between vector (p_from -> p_to) and the vertical axis.
    0 means perfectly vertical; 90 means horizontal.
    """

    vx = p_to.x - p_from.x
    vy = p_to.y - p_from.y
    norm = math.hypot(vx, vy)
    if norm < 1e-8:
        return float("nan")
    # vertical unit vector is (0, -1) or (0, 1) depending on axis;
    # we use absolute because camera y-axis direction varies across clients.
    cosv = abs(vy) / norm
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def point_line_signed_distance(p: Point, a: Point, b: Point) -> float:
    """
    Signed distance from p to line through a->b.
    Positive/negative depends on orientation; use abs() if unsure.
    """

    # Line in ax + by + c = 0
    A = b.y - a.y
    B = a.x - b.x
    C = -(A * a.x + B * a.y)
    denom = math.hypot(A, B)
    if denom < 1e-8:
        return float("nan")
    return (A * p.x + B * p.y + C) / denom


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

