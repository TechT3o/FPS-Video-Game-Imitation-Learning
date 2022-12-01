from data_recording.demonstration_recorder import DemoRecorder
from statics import start_countdown
from agent_training.transfer_learning import TransferLearner
from agent_playing_script import Agent
import win32api

WINDOW_COORDINATES = (0, 0, 1920, 1080)
RESET_CURSOR_FLAG = False
transfer_learner = TransferLearner()
recorder = DemoRecorder(window_coordinates=WINDOW_COORDINATES, reset_cursor_flag=RESET_CURSOR_FLAG,
                            save_path=transfer_learner.data_path)
print("S to start")
while 1:
    if win32api.GetAsyncKeyState(ord('S'))&0x0001 > 0:
        print('Starting to record in:')
        recorder.record_demo()

        model_path = transfer_learner.fine_tune()
        agent = Agent(model_path)

        print("Agent Ready!")
        input("Ready to test?")
        start_countdown(10)
        agent.run_agent()
        break
