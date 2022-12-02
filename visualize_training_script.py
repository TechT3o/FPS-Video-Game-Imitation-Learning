import pandas as pd
import matplotlib.pyplot as plt

path = "agent_5\\training.log"
k = 1

data = pd.read_csv(path)
plt.figure(1)
plt.title("Training curves")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(data["epoch"], data["val_loss"], label="validation_loss")
plt.plot(data["epoch"], data["loss"], label="training_loss")
plt.legend()
plt.show()