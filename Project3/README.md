# DS551/CS551/CS525 Inidividual Project 3
# Deep Q-learning Network(DQN)
## Setup
* Recommended programming IDE (integrated development environment): VS code (See [install VS code](https://code.visualstudio.com/)) 
* Install [Miniconda](https://www.python.org/downloads/)
* Create virtual environment and install Python 3: conda create -n myenv python=3.11.4. This will help you create a new conda environment named myenv. Gymnasium library supports for Python 3.8, 3.9, 3.10, 3.11 on Linux and macOS.
* Activate your virtual environment: `conda activate myenv`
* Install gymnasium: `pip install opencv-python-headless gymnasium[atari] autorom[accept-rom-license]` (See [install gymnasium](https://github.com/Farama-Foundation/Gymnasium))
* install pytorch: See [install pytorch](https://pytorch.org/get-started/locally/), pip install torch torchvision torchaudio
* For the  Atari wrapper, install the following two items: `pip install -U "ray[rllib]" ipywidgets`
* For successfully running code, you may also need to install the following item: `pip install --upgrade scipy numpy`.
* For video recording in testing, install the following three items: `pip install moviepy`, `pip install ffmpeg`.
* When testing, for nice output on the terminal, you need to install tqdm: `pip install tqdm`

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python main.py --test_dqn`

testing DQN while recording a video (recording video takes time, so usually you use this option when the number of testing episodes is small):
* `$ python main.py --test_dqn --record_video`  
