Hello!

This project can be used to teach neural network to recognize images (by default EMNIST data).

Data is missing, because it is too huge for the project. You should create new Data directory and put your data here.

There are several directories - in run_scripts there are scripts that you can directly run. There is constants.py with allfor all constants values. You should firstly run prepare_training_and_validation_data.py to preprocess and save preprocessed data. Then you can train your model using train.py. Then you can evaluate it with evaluate_results.py which draws graph or by looking at individual images with evaluate_draw_and_guess.py.

In EMNIST_classes there are python classes: data preparation class for preprocessing and data augmentation, model architecture with different pytorch modules, and module wich wraps model architecture class with usefull functions, e.g. train.