import numpy as np
import pytest

from rware.core.map_dsl import (
    parse_map_text, MapLayout,
    DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT, ALL_DIRS,
    WALL_NONE, WALL_TRANSPARENT, WALL_SOLID, move_bit,
)


def test_plain_text_without_sections_is_all_base():
    text = "p.p\npxp\np.p"
    layout = parse_map_text(text)
    assert layout.base.strip().split("\n") == ["p.p", "pxp", "p.p"]
    assert layout.walls.shape == (3, 3)
    assert (layout.walls == WALL_NONE).all()
    assert (layout.allowed_dirs == ALL_DIRS).all()


def test_base_and_overlay_sections():
    text = (
        "[base]\n"
        "p...p\n"
        "p.x.p\n"
        "p...p\n"
        "[overlay]\n"
        "..2..\n"
        ".#.W.\n"
        ".....\n"
    )
    layout = parse_map_text(text)
    assert layout.walls[1, 1] == WALL_TRANSPARENT
    assert layout.walls[1, 3] == WALL_SOLID
    assert layout.walls[0, 2] == WALL_NONE
    assert layout.allowed_dirs[0, 2] == DIR_RIGHT
    assert layout.allowed_dirs[2, 2] == ALL_DIRS


def test_hex_bitmask_decoding():
    text = "[base]\n...\n[overlay]\n3af"
    layout = parse_map_text(text)
    assert layout.allowed_dirs[0, 0] == (DIR_UP | DIR_RIGHT)      # 3
    assert layout.allowed_dirs[0, 1] == (DIR_RIGHT | DIR_LEFT)    # a
    assert layout.allowed_dirs[0, 2] == ALL_DIRS                  # f


def test_size_mismatch_raises():
    text = "[base]\n...\n...\n[overlay]\n...\n"
    with pytest.raises(ValueError, match="overlay"):
        parse_map_text(text)


def test_wall_on_station_cell_raises():
    # b (loadbox) under W must be rejected; same rule for g and w cells
    text = "[base]\n.b.\n[overlay]\n.W."
    with pytest.raises(ValueError, match="station"):
        parse_map_text(text)


def test_whitespace_and_indentation_stripped():
    text = "[base]\n   p . p  \n   p x p  \n[overlay]\n   . 2 .  \n   . . .  "
    layout = parse_map_text(text)
    assert layout.walls.shape == (2, 3)
    assert layout.allowed_dirs[0, 1] == DIR_RIGHT


def test_move_bit():
    assert move_bit(0, -1) == DIR_UP
    assert move_bit(1, 0) == DIR_RIGHT
    assert move_bit(0, 1) == DIR_DOWN
    assert move_bit(-1, 0) == DIR_LEFT
    assert move_bit(1, 1) == 0   # diagonal: never checked
    assert move_bit(0, 0) == 0
