# dreamer-pytorch
A PyTorch Implementation of [Dream to Control: Learning Behaviors by Latent Imagination][paper] by Danijar Hafner et.al.

Dreamer learns a world model that predicts ahead in a compact feature space. From imagined feature sequences, it learns a policy and state-value function. The value gradients are backpropagated through the multi-step predictions to efficiently learn a long-horizon policy.

### Original Project:
- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

This project uses some utilities taken from [PlaNet] (refactored for use in this project).


[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer
[PlaNet]: https://github.com/Kaixhin/PlaNet

### Usage
 - Run `main.py` for training.
 - Run `eval.py` for evaluation of a saved checkpoint.
 - Tensorboard will be used to display and store metrics and can be viewed by running the following:
 ```shell
 $ tensorboard --logdir <path_to_repository>/results
 ```
 - Visit tensorboard in your browser! By default tensorboard launches at `localhost:6006`. You might see a screen similar to this:
 ![Tensorboard](_assets/tensorboard.jpg)

### Results
The video on the **left** is the downscaled version of the **gym render**.  
The one on the **right** is **generated** by the decoder model.
#### During Training
![training](_assets/during_train.gif)

#### After Training
![training](_assets/trained.gif)


### Installation and running!
Install dependencies ...
- `pytorch==1.4.0`
- `tensorboard-pytorch==0.7.1`
- `tqdm==4.42.1`
- `torchvision==0.5.0`
- `gym==0.16.0`
