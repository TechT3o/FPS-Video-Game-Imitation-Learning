# Video-games-target-generalization

We are addressing the problem of training an agent how to perform a task. Usually such models need a large amount of data and leghty and computationally intensive training. In this project we first collect data of a human playing the sphere clash game mode of 3D aim trainer. Then, a recurrent convolutional neural network is built and trained on these data to learn how to shoot the targets. After, Dataset Aggregation (DAgger) can be done to collect more data to improve the agent's performance. Finally, few data of the agent playing the tile frenzy game mode are collected and the agent is fine tuned on these data to learn how to play the tile frenzy game mode effectively generalizing to a new environment.

Here is an example of the agent trained on 3 hours worth of data playing sphere clash:
[![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/87833804/207038595-80ddd5ec-9b82-4ebc-84ba-10c5006a0d4b.gif)](https://www.youtube.com/watchv=ZjJSiCquiA0)

Here is an example of the agent trained on 5 demonstrations playing tile frenzy:
[![ezgif com-gif-maker](https://user-images.githubusercontent.com/87833804/207038727-564c6f44-087b-4490-99a3-ba33afebef71.gif)](https://youtu.be/v13RcmPpCjM)


## Data collection

For the data collection [mouse_logger.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/data_recording/mouse_input.py) was created to record the mouse movement and whether the left click was pushed (shot or not). Frames from the computer screen are obtained by the get_image function in the [environment_axtraction.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/environment_extracting/environment_extraction.py). The mouse motion for every frame as well as the frame image path are recorded in a .csv file for every frame in [data_recording.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/data_recording/data_recording.py). To record data you must run [data_recording_main.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/data_recording_main.py) and set the save path and whether the first person shooter playing resets the game cursor in every frame which can be found by using the [mouse log test](https://github.com/TechT3o/Video-games-target-generalization/blob/main/data_recording/mouse_input.py#L143-L153) function

## Agent training

The option to build different types of models such as models with different bases, with LSTM layer or not and with or without the game features chain is written in [model_building.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/model_building.py). The data from the csvs are loaded and encoded appropriately to be fed in the neural network by [data_normalizing.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/data_normalizing.py).

This data can be fed in the training process either by loading everything in an array (requires a lot of RAM memory to load all of the data in a single array but model trains faster) using the class in [data_preprocessing.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/data_preprocessing.py) or by loading the batches of training or validation data for every epoch using a generator (does not require RAM memory but takes longer to train) using the class in [data_generator.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/data_generator.py). This can be selected by changing the generator option to 0 or 1 in [model_params.json](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/model_params.json).
The methods that train, evaluate and save the model, the callback functions and some training parameters are found in the ModelTrainer class in [model_training.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/model_training.py). A model can be trained by selecting the desired model building flags, training parameters and data and save paths and running [training_main.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/training_main.py).

A class that loads a pretrained agent and fine-tunes it with data stored (from a different game environment) in the one_shot_path parameter is found in [transfer_learning.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/transfer_learning.py). To do transfer learning simply run [transfer_learning.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/transfer_learning.py) after selecting the agent path and the new data path.

All of these flags, paths and training parameters can be adjusted by making changes to [model_params.json](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/model_params.json) which is loaded with the help of the singleton class Parameters in [parameters.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_training/parameters.py).

## Other scripts

The training of a model can be visualized with the training.log file produced when training by using the [visualize_training_script.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/visualize_training_script.py).
A trained agent can be loaded and used to take actions based on the captured screen frames by running [agent_playing_script.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/agent_playing_script.py). You can change how much the agent moves / shoots by changing the action and shooting thresholds in the class constructor and you can stop the agent from running by pressing the Q key.
To add the number of targets in the csv by color filtering first find the HSV threshold values by using the [color selection tool](https://github.com/TechT3o/Video-games-target-generalization/blob/main/environment_extracting/color_selection_tool.py), changing the lower_colour and upper_colour parameters in [environment_axtraction.py](https://github.com/TechT3o/Video-games-target-generalization/blob/main/environment_extracting/environment_extraction.py) and running the [target number csv addition script](https://github.com/TechT3o/Video-games-target-generalization/blob/main/target_number_csv_addition_script.py)

put singleton, statics, dagger, env extracting aimbot
