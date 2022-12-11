"""
Script where you put the path to teh training.log generated while training from the csv callback
and the training and validation loss curves are plotted.
"""

import pandas as pd
import matplotlib.pyplot as plt

path = "agents\\agent_14\\training.log"

data = pd.read_csv(path)
plt.figure(1)
plt.title("Training curves")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(data["epoch"], data["val_loss"], label="validation_loss")
plt.plot(data["epoch"], data["loss"], label="training_loss")
plt.legend()
plt.show()
