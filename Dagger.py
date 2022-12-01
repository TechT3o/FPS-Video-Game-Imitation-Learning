from data_recording.data_recorder import DataRecorder
from agent_playing_script import Agent
from agent_training.parameters import Parameters
from statics import start_countdown, check_and_create_directory
from typing import Tuple
import os


class Dagger:
    def __init__(self, reset_cursor_flag: bool = False, window_coordinates: Tuple[int, int, int, int] = (0, 0, 1920, 1080)):

        self.params = Parameters()
        self.agent_path = self.params.agent_path
        self.dagger_path = self.params.dagger_path
        check_and_create_directory(self.dagger_path)

        self.recorder = DataRecorder(window_coordinates=window_coordinates,
                                     reset_cursor_flag=reset_cursor_flag, save_path=self.dagger_path)
        self.agent = Agent(os.path.join(self.agent_path, 'agent.h5'))

    def run_dagger(self):
        start_countdown(3)
        while True:
            # press Q to let expert take control
            print('agent playing')
            self.agent.run_agent()
            print('expert showing')
            self.recorder.run()


if __name__ == "__main__":
    dagger = Dagger()
    dagger.run_dagger()


