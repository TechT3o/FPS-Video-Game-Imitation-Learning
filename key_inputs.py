# adapted from https://github.com/Sentdex/pygta5/blob/master/getkeys.py

import win32api
import time


class MouseLogger:
    previous_status_r: int
    previous_status_l: int
    cursor_x: int
    cursor_y: int

    def __init__(self):
        # TODO put action space and discretization somewhere around here?
        self.action_state_boundaries = None
        self.cursor_x, self.cursor_y = win32api.GetCursorPos()
        self.held_down_l = 0
        self.clicked_l = 0
        self.held_down_r = 0
        self.clicked_r = 0
        self.fps = 20
        self.previous_status_l = win32api.GetKeyState(0x01)
        self.previous_status_r = win32api.GetKeyState(0x02)

    def _mouse_l_click_check(self) -> int:
        self.held_down_l = 0
        self.clicked_l = 0
        current_status = win32api.GetKeyState(0x01)
        if current_status < 0:
            self.held_down_l = 1  # held down click
            if current_status != self.previous_status_l and not self.previous_status_l < 0:
                self.clicked_l = 1  # just tapped this
        return current_status

    def _mouse_r_click_check(self) -> int:
        self.held_down_r = 0
        self.clicked_r = 0
        current_status = win32api.GetKeyState(0x02)

        if current_status < 0:
            self.held_down_r = 1  # held down click
            if current_status != self.previous_status_r and not self.previous_status_r < 0:
                self.clicked_r = 1  # just tapped this
        return current_status

    def mouse_log_test(self) -> None:
        # TODO: figure how to get values outside of while loop. Should it be read inside main or run at different Thread
        while True:
            loop_start_time = time.time()  # this is in seconds

            current_status_l = self._mouse_l_click_check()
            current_status_r = self._mouse_r_click_check()
            print('l_click', self.clicked_l, ' l_held', self.held_down_l, ' | r_click', self.clicked_r,
                  ' r_held', self.held_down_r)
            self.cursor_x, self.cursor_y = win32api.GetCursorPos()
            self.previous_status_l = current_status_l
            self.previous_status_r = current_status_r
            # time.sleep(0.1)

            # wait until end of time step
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


if __name__ == "__main__":
    logger = MouseLogger()
    logger.mouse_log_test()
    print(f'cursor x coord {logger.x_coord} and y coord {logger.y_coord}')
