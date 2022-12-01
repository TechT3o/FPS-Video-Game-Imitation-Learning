import tensorflow as tf
import numpy as np
from statics import start_countdown, mouse_action, preprocess_image
import time
import mss
from win32api import GetAsyncKeyState


lil_portillo = tf.keras.models.load_model('agent_3/agent.h5')
print(lil_portillo.summary())
ACTION_SPACE_X = np.array([-300, -200, -150, -100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 50, 100, 150, 200, 300])
ACTION_SPACE_Y = np.array([-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100])

FPS = 24
start_countdown(6)
buffer = []

while True:

    loop_start_time = time.time()
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        img = np.array(sct.grab(monitor))[:, :, :3]

    processed_image = preprocess_image(img, (240, 135)).reshape((240, 135, 3))
    buffer.append(processed_image)

    if len(buffer) >= 15:
        features, click_pred, mouse_x_pred, mouse_y_pred = lil_portillo.predict_on_batch(np.array([buffer]))
        buffer.pop(0)
        x_predictions = mouse_x_pred[0][-1]
        y_predictions = mouse_y_pred[0][-1]
        click_predictions = click_pred[0][-1]

        # print(x_predictions)

        x_motion = ACTION_SPACE_X[x_predictions.argmax()]
        y_motion = ACTION_SPACE_Y[y_predictions.argmax()]
        click = 0 if click_predictions.argmax() == 0 else 1

        mouse_action(x_motion, y_motion, click, 0.01)

    if GetAsyncKeyState(ord('Q')):
        print('Quit')
        break

    while time.time() < loop_start_time + 1 / FPS:
        pass
    print(1 / (time.time() - loop_start_time))