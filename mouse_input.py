# adapted from https://github.com/Sentdex/pygta5/blob/master/getkeys.py

import win32api
import time
from typing import Tuple


class MouseLogger:
    """
    Class that logs the mouse inputs
    """
    previous_status_r: int
    previous_status_l: int
    cursor_x: int
    cursor_y: int
    previous_cursor_x: int
    previous_cursor_x: int
    delta_x: int
    delta_y: int
    screen_center: Tuple[int, int]
    game_resets_cursor: bool

    def __init__(self, window_coordinates: Tuple[int, int] = (1920/2, 1080/2), reset_cursor_flag: bool = True):
        """
        class constructor
        :param window_coordinates: coordinates of the screen
        :param reset_cursor_flag: whether the game trying to log resets the cursor to the center value after every frame
        """
        self.action_state_boundaries = None
        self.cursor_x, self.cursor_y = win32api.GetCursorPos()
        self.held_down_l = 0
        self.clicked_l = 0
        self.held_down_r = 0
        self.clicked_r = 0
        self.fps = 20
        self.previous_status_l = win32api.GetKeyState(0x01)
        self.previous_status_r = win32api.GetKeyState(0x02)
        self.screen_center = window_coordinates
        self.previous_cursor_x, self.previous_cursor_y = win32api.GetCursorPos()
        self.delta_x = self.cursor_x - self.previous_cursor_x
        self.delta_y = self.cursor_y - self.previous_cursor_y
        self.game_resets_cursor = reset_cursor_flag

    def _mouse_l_click_check(self) -> int:
        """
        Checks if the left side of the mouse was clicked or is being held down
        :return: current mouse status
        """
        self.held_down_l = 0
        self.clicked_l = 0
        current_status = win32api.GetKeyState(0x01)
        if current_status < 0:
            self.held_down_l = 1  # held down click
            if current_status != self.previous_status_l and not self.previous_status_l < 0:
                self.clicked_l = 1  # just tapped this
        return current_status

    def _mouse_r_click_check(self) -> int:
        """
        Checks if the right side of the mouse was clicked or is being held down
        :return: current mouse status
        """
        self.held_down_r = 0
        self.clicked_r = 0
        current_status = win32api.GetKeyState(0x02)

        if current_status < 0:
            self.held_down_r = 1  # held down click
            if current_status != self.previous_status_r and not self.previous_status_r < 0:
                self.clicked_r = 1  # just tapped this
        return current_status

    def cursor_reset(self) -> None:
        """
        Checks if cursor was reset and tries to ignore this motion
        :return: None
        """
        # Does not work as intended because cursor reading stays at constant at edges of screen
        # if not (0 < self.cursor_x < 1919):
        #     self.cursor_x = self.previous_cursor_x
        # if not (0 < self.cursor_y < 1079):
        #     self.cursor_y = self.previous_cursor_y

        # effort to fix reset motion delta x detected after every movement but found that for these games
        # it is better to just keep previous cursor as the center of the screen
        if (self.cursor_x == self.screen_center[0] and self.cursor_y == self.screen_center[1]) and \
                ((abs(self.cursor_x - self.previous_cursor_x) > 0) or
                 (abs(self.cursor_y - self. previous_cursor_y > 0))):
            print('reset cursor')
            self.previous_cursor_x = self.cursor_x
            self.previous_cursor_y = self.cursor_y

    def get_mouse_states(self) -> None:
        """
        Gets the state of the mouse cursor, its final motion and the status of the left and right clicks
        :return: None
        """

        current_status_l = self._mouse_l_click_check()
        current_status_r = self._mouse_r_click_check()
        self.cursor_x, self.cursor_y = win32api.GetCursorPos()
        # self.cursor_reset()
        self.delta_x = self.cursor_x - self.previous_cursor_x
        self.delta_y = self.cursor_y - self.previous_cursor_y

        # print('l_click', self.clicked_l, ' l_held', self.held_down_l, ' | r_click', self.clicked_r,
        #       ' r_held', self.held_down_r)
        # print(f'cursor x coord {self.cursor_x} and y coord {self.cursor_y}')
        # print(f'delta_x is {self.delta_x} and delta_y is {self.delta_y}')
        self.previous_status_l = current_status_l
        self.previous_status_r = current_status_r

        if self.game_resets_cursor:
            self.previous_cursor_x = self.screen_center[0]
            self.previous_cursor_y = self.screen_center[1]

        else:
            self.previous_cursor_x = self.cursor_x
            self.previous_cursor_y = self.cursor_y

    def mouse_log_test(self) -> None:
        """
        Used to test the mouse behaviour
        :return: None
        """
        while True:
            loop_start_time = time.time()
            self.get_mouse_states()
            while time.time() < loop_start_time + 1/self.fps:
                pass

    @property
    def x_coord(self):
        return self.cursor_x

    @property
    def y_coord(self):
        return self.cursor_y

    @property
    def l_click(self):
        return self.clicked_l

    @property
    def r_click(self):
        return self.clicked_r

    @property
    def l_held(self):
        return self.held_down_l

    @property
    def r_held(self):
        return self.held_down_r

    @property
    def d_x(self):
        return self.delta_x

    @property
    def d_y(self):
        return self.delta_y


if __name__ == "__main__":
    logger = MouseLogger()
    logger.mouse_log_test()
