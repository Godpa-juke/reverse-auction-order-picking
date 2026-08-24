"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""
_TICKPERTIME = 1.5

import math
import os
import sys

import numpy as np
import math
import six
from gymnasium import error
from rware.core import Direction, State
# from rware.warehouse_copy import Direction, State

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )

# 설정 매크로 변수 선언
RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_MARGENTA = (244,0,220)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_NO_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_HUMAN_COLOR = _TEAL
_AGENT_LOADED_COLOR = _RED
_HUMAN_PICKING_COLOR = _MARGENTA
_ROBOT_PICKING_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_ROBOT_BOX_LOAD_COLOR = _ORANGE
_GOAL_COLOR = (60, 60, 60)
_WORK_COLOR = (100,200,100)
_WAIT_COLOR = (0,200,200)
_LOADBOX_COLOR = (255,100,100)
_WALL_SOLID_COLOR = (100, 100, 100)
_WALL_TRANSPARENT_COLOR = (150, 190, 230)
_SHELF_PADDING = 2


_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_SPOTS  = 2
_LAYER_HUMAN  = 3

_STRATEGY_ABBREVIATIONS = {
    "nearest_robot_first": "NRF",
    "first_robot_arrived": "FRA",
    "shortest_service_robot": "SSR",
    "nearest_idle": "NI",
    "legacy_batch": "LB",
    "nearest_idle_within_zone": "NI",
}


def _strategy_abbreviation(name: str) -> str:
    if not name:
        return "N/A"
    return _STRATEGY_ABBREVIATIONS.get(name.lower(), name.upper())

# 디스플레이 정보 취득 함수
def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )

# 이미지 출력 클래스
class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.grid_size = 12
        self.grid_size = 6
        self.icon_size = 20
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            display=display,
            caption='Order Picking Simulator',
        )
        icon = pyglet.image.load('rware/Img/ICON.png')
        self.window.set_icon(icon)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        self.shelf_batch = None
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # 종료 메소드
    def close(self):
        self.window.close()

    # 사용자 종료 메소드
    def window_closed_by_user(self):
        self.isopen = False
        exit()

    # 범위 지정 메소드
    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    # 렌더링 메소드
    def render(self, env, return_rgb_array=False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        strategy_name = getattr(env, "human_assignment_strategy", "")
        caption = f"Order Picking Simulator [{_strategy_abbreviation(strategy_name)}]" if strategy_name else "Order Picking Simulator"
        self.window.set_caption(caption)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # self._draw_grid()
        self._draw_goals(env)
        self._draw_walls(env)
        self._draw_shelfs(env)
        self._draw_agents(env)
        # self._draw_obj_ids(env)
        self._draw_timer_and_completed_order(env)
        # self._draw_AGV_OPrate(env)
        # self._draw_Station_OPrate(env)
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    # 그리드 드로잉 메소드
    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # VERTICAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    # 랙 드로잉 메소드
    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed

            shelf_color = (
                _SHELF_NO_REQ_COLOR if shelf.id in env.not_used_shelf else _SHELF_COLOR
            )

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        # (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # TL - X
                        # (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TL - Y
                        # (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # TR - X
                        # (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TR - Y
                        # (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # BR - X
                        # (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BR - Y
                        # (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # BL - X
                        # (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BL - Y
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * shelf_color),
            )
        batch.draw()

    # 작업 큐 드로잉 메소드
    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,    # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _GOAL_COLOR),
            )
        batch.draw()

        for picking in env.picking_queue:
            x, y = picking
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _WORK_COLOR),
            )
        batch.draw()


        for loadbox in env.loadbox_queue:
            x, y = loadbox
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _LOADBOX_COLOR),
            )
        batch.draw()
        for wait in env.wait_queue:
            x, y = wait
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _WAIT_COLOR),
            )
        batch.draw()

        # for path_list in env.path_list:
        #     for path in path_list:
        #         x, y = path[0][1], path[0][0]
        #         y = self.rows - y - 1  # pyglet rendering is reversed
        #         batch.add(
        #             4,
        #             gl.GL_QUADS,
        #             None,
        #             (
        #                 "v2f",
        #                 (
        #                     (self.grid_size + 1) * x + 1,  # TL - X
        #                     (self.grid_size + 1) * y + 1,  # TL - Y
        #                     (self.grid_size + 1) * (x + 1),  # TR - X
        #                     (self.grid_size + 1) * y + 1,  # TR - Y
        #                     (self.grid_size + 1) * (x + 1),  # BR - X
        #                     (self.grid_size + 1) * (y + 1),  # BR - Y
        #                     (self.grid_size + 1) * x + 1,  # BL - X
        #                     (self.grid_size + 1) * (y + 1),  # BL - Y
        #                 ),
        #             ),
        #             ("c3B", 4 * _ORANGE),
        #         )
        #     batch.draw()
        #
        # env.path_list = []

    # 벽(월) 드로잉 메소드 -- map-DSL overlay walls (Task 7)
    def _draw_walls(self, env):
        walls = getattr(env, "walls", None)
        if walls is None:
            return
        batch = pyglet.graphics.Batch()

        grid_h, grid_w = walls.shape
        for y in range(grid_h):
            for x in range(grid_w):
                wtype = walls[y, x]
                if not wtype:
                    continue
                wall_color = (
                    _WALL_SOLID_COLOR if wtype == 2 else _WALL_TRANSPARENT_COLOR
                )
                ry = self.rows - y - 1  # pyglet rendering is reversed
                batch.add(
                    4,
                    gl.GL_QUADS,
                    None,
                    (
                        "v2f",
                        (
                            (self.grid_size + 1) * x + 1,  # TL - X
                            (self.grid_size + 1) * ry + 1,  # TL - Y
                            (self.grid_size + 1) * (x + 1),  # TR - X
                            (self.grid_size + 1) * ry + 1,  # TR - Y
                            (self.grid_size + 1) * (x + 1),  # BR - X
                            (self.grid_size + 1) * (ry + 1),  # BR - Y
                            (self.grid_size + 1) * x + 1,  # BL - X
                            (self.grid_size + 1) * (ry + 1),  # BL - Y
                        ),
                    ),
                    ("c3B", 4 * wall_color),
                )
        batch.draw()

    # 로봇 드로잉 메소드
    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        # Need to Modify by Jw.son 2022.07.23
        # we need to consider KIVA HW Shape
        # radius = self.grid_size / 3

        resolution = 6
        loop_cnt = 3

        for loop_idx in range(2,loop_cnt):
            for agent in reversed(env.agents):
                if agent.id not in env.agent_id_list: continue
                # col, row = agent.x, agent.y
                col = agent.prev_x + (agent.x - agent.prev_x)*((loop_idx*5)/10)
                row = agent.prev_y + (agent.y - agent.prev_y)*((loop_idx*5)/10)

                row = self.rows - row - 1  # pyglet rendering is reversed

                if agent.agent_type == True:
                    radius = self.grid_size / 4 + 4
                    resolution = 8
                else:
                    radius = self.grid_size / 3 + 5
                    resolution = 6

                # make a circle
                verts = []
                for i in range(resolution):
                    angle = 2 * math.pi * i / resolution
                    x = (
                        radius * math.cos(angle)
                        + (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                    )
                    y = (
                        radius * math.sin(angle)
                        + (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                    )
                    verts += [x, y]
                circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
                # draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
                draw_color = _HUMAN_COLOR if agent.agent_type == True else _AGENT_COLOR

                if agent.agent_type == True and (
                    str(agent.state) == str(State.HUMAN_PICKING) or str(agent.state) == str(
                    State.HUMAN_DONE)): draw_color = _HUMAN_PICKING_COLOR
                if agent.agent_type == False and (
                    str(agent.state) == str(State.ROBOT_PICKING) or str(agent.state) == str(
                    State.ROBOT_DROP) or str(agent.state) == str(State.ROBOT_LOAD)): draw_color = _ROBOT_PICKING_COLOR

                # if agent.agent_type == False and (
                #     str(agent.state) == str(State.ROBOT_LOAD) or str(agent.state) == str(
                #     State.ROBOT_LOAD)): draw_color = _ROBOT_BOX_LOAD_COLOR

                glColor3ub(*draw_color)
                circle.draw(GL_POLYGON)

            for agent in env.agents:
                if agent.id not in env.agent_id_list: continue
                # col, row = agent.x, agent.y
                col = agent.prev_x + (agent.x - agent.prev_x) * ((loop_idx * 5) / 10)
                row = agent.prev_y + (agent.y - agent.prev_y) * ((loop_idx * 5) / 10)
                row = self.rows - row - 1  # pyglet rendering is reversed
                radius = self.grid_size / 4 + 5 if agent.agent_type == True else self.grid_size / 3 + 5
                # Modified by Jw.son 2022.07.23
                # Agent's direction display is modified by rectangle shape

                if agent.agent_type == False:
                    diag_param = 0
                    dir_radius = radius * 8 / 10
                    dir_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1 \
                            + (dir_radius - 1 if agent.dir.value == Direction.RIGHT.value else 0) \
                            + (dir_radius - 6 if agent.dir.value == Direction.UPRIGHT.value else 0) \
                            + (dir_radius - 6 if agent.dir.value == Direction.DOWNRIGHT.value else 0) \
                            + (-dir_radius + 1 if agent.dir.value == Direction.LEFT.value else 0) \
                            + (-dir_radius + 6 if agent.dir.value == Direction.UPLEFT.value else 0) \
                            + (-dir_radius + 6 if agent.dir.value == Direction.DOWNLEFT.value else 0)

                    dir_y = (self.grid_size + 1) * row + self.grid_size // 2 + 1 \
                            + (dir_radius - 3 if agent.dir.value == Direction.UP.value else 0) \
                            + (dir_radius - 8 if agent.dir.value == Direction.UPRIGHT.value else 0) \
                            + (dir_radius - 8 if agent.dir.value == Direction.UPLEFT.value else 0) \
                            + (-dir_radius + 2 if agent.dir.value == Direction.DOWN.value else 0) \
                            + (-dir_radius + 8 if agent.dir.value == Direction.DOWNRIGHT.value else 0) \
                            + (-dir_radius + 8 if agent.dir.value == Direction.DOWNLEFT.value else 0)

                else:
                    diag_param = 0
                    dir_radius = radius * 7 / 10
                    dir_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1 \
                            + (dir_radius - 1 if agent.dir.value == Direction.RIGHT.value else 0) \
                            + (dir_radius - 6 if agent.dir.value == Direction.UPRIGHT.value else 0) \
                            + (dir_radius - 6 if agent.dir.value == Direction.DOWNRIGHT.value else 0) \
                            + (-dir_radius + 1 if agent.dir.value == Direction.LEFT.value else 0) \
                            + (-dir_radius + 6 if agent.dir.value == Direction.UPLEFT.value else 0) \
                            + (-dir_radius + 6 if agent.dir.value == Direction.DOWNLEFT.value else 0)

                    dir_y = (self.grid_size + 1) * row + self.grid_size // 2 + 1 \
                            + (dir_radius - 3 if agent.dir.value == Direction.UP.value else 0) \
                            + (dir_radius - 8 if agent.dir.value == Direction.UPRIGHT.value else 0) \
                            + (dir_radius - 8 if agent.dir.value == Direction.UPLEFT.value else 0) \
                            + (-dir_radius + 2 if agent.dir.value == Direction.DOWN.value else 0) \
                            + (-dir_radius + 8 if agent.dir.value == Direction.DOWNRIGHT.value else 0) \
                            + (-dir_radius + 8 if agent.dir.value == Direction.DOWNLEFT.value else 0)

                black_circle = pyglet.shapes.Circle(dir_x, dir_y, 4, color=(0, 0, 0), batch=batch)
                black_circle.draw()
                # red_sequare.draw()
            batch.draw()


    # 타이머 및 완료 주문 수 드로잉 메소드
    def _draw_timer_and_completed_order(self, env):
        cur_time = env.internal_timer

        total_human_distance = [env.agents[idx].total_distance for idx in range(env.n_humans)]
        total_robot_distance = [env.agents[idx].total_distance for idx in range(env.n_humans,env.n_agents)]
        total_human_distance.sort()
        total_robot_distance.sort()

        running_human_cnt = env.running_human_cnt
        running_robot_cnt = env.running_robot_cnt

        cur_hour = (cur_time) // 3600
        cur_min  = ((cur_time) % 3600) // 60
        cur_sec  = (((cur_time) % 3600) % 60)

        str_batch = pyglet.graphics.Batch()
        strategy_name = getattr(env, "human_assignment_strategy", "")
        strategy_abbrev = _strategy_abbreviation(strategy_name)
        cur_strategy_str = f'STRATEGY : {strategy_abbrev}'
        Label0 = pyglet.text.Label(
            cur_strategy_str,
            font_name='Aerial',
            font_size=13,
            bold=True,
            x=(self.grid_size + 1) * self.cols - 100 - 4 - 5,
            y=(self.grid_size + 1) * self.rows - 150,
            anchor_x='center',
            anchor_y='center',
            color=(0, 0, 0, 255),
            batch=str_batch,
        )
        completed_batch = env.completed_batch
        all_of_completed_order = env.all_of_completed_order

        cur_time_str = str('TICK : {0:02d} : {1:02d} : {2:02d}'.format(cur_hour,cur_min,cur_sec))
        Label1 = pyglet.text.Label(cur_time_str, font_name='Aerial', font_size=13, bold=True,
                                   x=(self.grid_size + 1) * self.cols-100-4-5, y=(self.grid_size + 1) * self.rows - 170,
                                  anchor_x='center', anchor_y='center', color = (0, 0, 0, 255), batch=str_batch)

        cur_completed_batch_str = str('Batch Done : {:04d}'.format(completed_batch))
        Label2 = pyglet.text.Label(cur_completed_batch_str, font_name='Aerial', font_size=13, bold=True,
                                   x=(self.grid_size + 1) * self.cols-100-4-5, y=(self.grid_size + 1) * self.rows - 190,
                                   anchor_x='center', anchor_y='center', color = (0, 0, 0, 255), batch=str_batch)

        cur_all_off_completed_order_str = str('  SKU Done : {:04d}'.format(all_of_completed_order))
        Label3 = pyglet.text.Label(cur_all_off_completed_order_str, font_name='Aerial', font_size=13, bold=True,
                                   x=(self.grid_size + 1) * self.cols - 100-2-5,
                                   y=(self.grid_size + 1) * self.rows - 210,
                                   anchor_x='center', anchor_y='center', color=(0, 0, 0, 255), batch=str_batch)

        str_batch.draw()
    # AGV 동작율 드로잉 메소드
    def _draw_AGV_OPrate(self, env):
        entire_n_agent = env.n_agents
        current_n_agent = env.using_agent
        op_rate = int((current_n_agent/entire_n_agent)*100)
        cur_time_str = str('AGV OP : {:02d}%'.format(op_rate))
        Label = pyglet.text.Label(cur_time_str, font_name='Aerial', font_size=8, bold=True,
                                  # x=(self.grid_size + 1) * self.cols-50, y=(self.grid_size + 1) * self.rows-125,
                                  x=(self.grid_size + 1) * self.cols - 35, y=40,

                                  anchor_x='center', anchor_y='center')


        Label.color = (0, 0, 0, 255)
        # Label.draw()

    # 스테이션 가동율 드로잉 메소드
    def _draw_Station_OPrate(self, env):
        # using_station1 = env.using_station[0]
        # using_station2 = env.using_station[1]
        # internal_timer = env.internal_timer
        #
        # if internal_timer == 0:
        #     op_rate1 = 0
        #     op_rate2 = 0
        #
        # else:
        #     op_rate1 = int((using_station1 / internal_timer) * 100)
        #     op_rate2 = int((using_station2 / internal_timer) * 100)
        #
        # cur_str1 = str('St1 OP : {:02d}%'.format(op_rate1))
        # Label1 = pyglet.text.Label(cur_str1, font_name='Aerial', font_size=8, bold=True,
        #                            # x=(self.grid_size + 1) * self.cols-50, y=(self.grid_size + 1) * self.rows-175,
        #                            x=(self.grid_size + 1) * self.cols-35, y=25,
        #                            anchor_x='center', anchor_y='center')
        #
        # cur_str2 = str('St2 OP : {:02d}%'.format(op_rate2))
        # Label2 = pyglet.text.Label(cur_str2, font_name='Aerial', font_size=8, bold=True,
        #                            # x=(self.grid_size + 1) * self.cols - 50, y=(self.grid_size + 1) * self.rows - 225, # 225
        #                            x=(self.grid_size + 1) * self.cols-35, y=10,   # 225
        #                            anchor_x='center', anchor_y='center')

        cur_str1 = str('CJ Logistics')
        Label1 = pyglet.text.Label(cur_str1, font_name='Aerial', font_size=10, bold=True,
                                   # x=(self.grid_size + 1) * self.cols-50, y=(self.grid_size + 1) * self.rows-175,
                                   x=(self.grid_size + 1) * self.cols - 50, y=50,
                                   anchor_x='center', anchor_y='center')

        cur_str2 = str('Robot Control Team Lois APRIL')
        Label2 = pyglet.text.Label(cur_str2, font_name='Aerial', font_size=10, bold=True,
                                   # x=(self.grid_size + 1) * self.cols-50, y=(self.grid_size + 1) * self.rows-175,
                                   x=(self.grid_size + 1) * self.cols - 120, y=30,
                                   anchor_x='center', anchor_y='center')

        cur_str3 = str('Made by Jw.son, Jw.Sung')
        Label3 = pyglet.text.Label(cur_str3, font_name='Aerial', font_size=10, bold=True,
                                   # x=(self.grid_size + 1) * self.cols - 50, y=(self.grid_size + 1) * self.rows - 225, # 225
                                   x=(self.grid_size + 1) * self.cols - 100, y=10,  # 225
                                   anchor_x='center', anchor_y='center')

        Label1.color = (0, 0, 0, 255)
        Label2.color = (0, 0, 0, 255)
        Label3.color = (0, 0, 0, 255)

        Label1.draw()
        Label2.draw()
        Label3.draw()



    # Made by Jw.son 2022.07.23
    # Agent, Shelf ID Draw Method Define
    # 오브젝트 ID 정보 드로잉 메소드
    def _draw_obj_ids(self, env):

        robot_batch = pyglet.graphics.Batch()
        for agent in env.agents:
            if agent.id not in env.agent_id_list: continue
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            cent_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            cent_y = (self.grid_size + 1) * row + self.grid_size // 2 + 5

            if agent.agent_type == False:
                cent_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1
                cent_y = (self.grid_size + 1) * row + self.grid_size // 2 + 6
            else:
                cent_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1
                cent_y = (self.grid_size + 1) * row + self.grid_size // 2 - 2
            agent_ID = str(agent.id)

            Label = pyglet.text.Label(agent_ID, font_name='Aerial', font_size=5, bold=True, x=cent_x, y=cent_y,
                                      anchor_x='center', anchor_y='center', color=(0, 0, 0, 255), batch=robot_batch)
        robot_batch.draw()

        if self.shelf_batch is None:
            self.shelf_batch = pyglet.graphics.Batch()
            for shelf in env.shelfs:
                x, y = shelf.x, shelf.y
                y = self.rows - y - 1  # pyglet rendering is reversed
                shelf_color = (
                    _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR
                )

                cent_x = (self.grid_size + 1) * x + self.grid_size // 2 + 1
                cent_y = (self.grid_size + 1) * y + self.grid_size // 2
                shelf_ID = str(shelf.id)

                Label = pyglet.text.Label(shelf_ID, font_name='Aerial', font_size=5, bold=False, x=cent_x, y=cent_y,
                                          anchor_x='center', anchor_y='center',color = (255, 255, 255, 255),batch=self.shelf_batch)

        self.shelf_batch.draw()

    # 배지 정보 드로잉 메소드
    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = (self.grid_size / 5)

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()
