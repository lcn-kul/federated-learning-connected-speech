# Federated Learning for Connected Speech
Federated learning for the classification of various neurodegenerative diseases based on transcribed speech data from continuous speech.

![INCLUDE_flwr_connected_speech](https://github.com/lcn-kul/federated-learning-connected-speech/assets/79714136/70dda522-95b4-42cf-a7f6-d724529c77c3)

## Prerequisites 
* Git (e.g., [GitHub Desktop](https://desktop.github.com/))
* Python >=3.8 (e.g., Python 3.11)
* A python virtual environment manager
* An server_details.env file with the server address and port (shared beforehand)

The easiest way to get Python and a virtual environment manager is by installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

## Installation & Setup
1. Clone the repository to get the code
   * [GitHub Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop)
   * [Command line](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
2. Create a virtual environment using the ```requirements.txt``` file
   * Either open a terminal or use the Anaconda prompt (Windows) and in the project directory (```your-path/federated-learning-connected-speech```) create your environment (e.g., ```conda create -n fl-cs python=3.11```) and install the requirements (e.g, ```pip install -r requirements.txt```)
3. Add your private files to the project (they will not be uploaded to GitHub)
   * Add the ```server_details.env``` file with the server address and port (shared beforehand) in the project directory (```your-path/federated-learning-connected-speech```)
   * Add your disease-group specific subject files to the respective sub-folders in ```your-path/federated-learning-connected-speech/data/input```. Each subject should have a separate ```.txt``` file, the file names do not matter as long as they do not contain the terms "train"/"test"/"val".

## Starting the Client
1. Open a terminal or use the Anaconda prompt (Windows) and activate your virtual environment (e.g., ```conda activate fl-cs```)
2. Navigate to the folder in which the ```client.py``` script is located (e.g., ```cd your-path/federated-learning-connected-speech/src/fl_connected_speech```)
3. Start the client (e.g., ```python client.py```). **A log file will be created automatically in data/output/.**

Ideally, the client should be started in the background, as it will need to stay running until the federated learning rounds are completed (e.g., through ```nohup``` or ```screen``` on Linux, possibly ```start``` on Windows).

If the client is started successfully, it will print something like this:
```
DEBUG flwr 2024-01-08 19:23:10,663 | connection.py:42 | ChannelConnectivity.IDLE
DEBUG flwr 2024-01-08 19:23:10,665 | connection.py:42 | ChannelConnectivity.CONNECTING
DEBUG flwr 2024-01-08 19:23:10,668 | connection.py:42 | ChannelConnectivity.READY
```
Once all federated learning rounds are completed, the client will print something like this and the program will terminate:
```
Training Finished.
{'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
DEBUG flwr 2024-01-08 19:21:48,454 | connection.py:141 | gRPC channel closed
INFO flwr 2024-01-08 19:21:48,454 | app.py:304 | Disconnect and shut down
```

In case it cannot connect to the server or if the server is not running, it will print something like this:
```
grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
... 
Failed to connect to remote host: Connection refused
```
