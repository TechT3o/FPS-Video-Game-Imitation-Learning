from data_recording.data_recorder import DataRecorder
from statics import start_countdown
from agent_training.transfer_learning import TransferLearner


WINDOW_COORDINATES = (0, 0, 1920, 1080)
RESET_CURSOR_FLAG = False


start_countdown(10)
transfer_learner = TransferLearner()
recorder = DataRecorder(window_coordinates=WINDOW_COORDINATES, reset_cursor_flag=RESET_CURSOR_FLAG,
                        save_path=transfer_learner.data_path)

recorder.run()

transfer_learner.fine_tune()

print("Agent Ready!")

input("Ready to test?")

start_countdown(10)