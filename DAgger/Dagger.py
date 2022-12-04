from data_recording.data_recorder import DataRecorder
from agent_playing_script import Agent
from agent_training.parameters import Parameters
from statics import start_countdown, check_and_create_directory
from typing import Tuple
from agent_training.transfer_learning import TransferLearner
from agent_training.data_preprocessing import DataProcessor
import tensorflow as tf


class Dagger(TransferLearner):
    def __init__(self, reset_cursor_flag: bool = False, window_coordinates: Tuple[int, int, int, int] = (0, 0, 1920, 1080)):

        super(Dagger,self).__init__()

        self.dagger_path = self.params.dagger_path
        check_and_create_directory(self.dagger_path)

        self.recorder = DataRecorder(window_coordinates=window_coordinates,
                                     reset_cursor_flag=reset_cursor_flag, save_path=self.dagger_path)
        self.agent = Agent(self.agent_path)

    def run_dagger(self):
        start_countdown(3)
        while True:
            # press Q to let expert take control
            print('agent playing')
            self.agent.run_agent()
            print('expert showing')
            self.recorder.run()

    def train_dagger(self):
        self.compile_agent
        dataset = DataProcessor(self.dagger_path)
        batch_size = 4

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("new_agent.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger('training.log'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=20, verbose=1,
                                             mode="min")]
        print("Compiling model")
        self.compile_agent()
        self.model.fit(x=dataset.x_train, y=dataset.y_train, batch_size=batch_size, epochs=50000, callbacks=callbacks,
                                              validation_data=(dataset.x_val, dataset.y_val),
                                              steps_per_epoch=dataset.x_train.shape[0] // batch_size,
                                              validation_steps=dataset.x_val.shape[0] // batch_size,
                                              verbose=1)


if __name__ == "__main__":
    dagger = Dagger()
    dagger.run_dagger()


