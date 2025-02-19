# Federated Learning for Connected Speech
Federated learning for the classification of various neurodegenerative diseases based on transcribed speech data from continuous speech.

![Overview_Figure_Multilingual_Connected_Speech_v1](https://github.com/user-attachments/assets/65bff8b1-9352-4beb-935b-a58f693a6e88)

## Prerequisites 
* Git (e.g., [GitHub Desktop](https://desktop.github.com/))
* Python >=3.8 (e.g., Python 3.11)
* A python virtual environment manager
* An server_details.env file with the server address and port (shared beforehand, not needed for external evaluators)

The easiest way to get Python and a virtual environment manager is by installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

## Installation & Setup
1. Clone the repository to get the code
   * [GitHub Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop)
   * [Command line](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
2. Create a virtual environment using the ```requirements.txt``` file
   * Either open a terminal or use the Anaconda prompt (Windows) and in the project directory (```your-path/federated-learning-connected-speech```) create your environment (e.g., ```conda create -n fl-cs python=3.11```) and install the requirements (e.g, ```pip install -r requirements.txt```)
3. Add your private files to the project (they will not be uploaded to GitHub)
   * **For clients involved in training the federated learning model only**: Add the ```server_details.env``` file with the server address and port (shared beforehand) in the project directory (```your-path/federated-learning-connected-speech```)
   * **For clients and external evaluators**: Add your disease-group specific subject files to the respective sub-folders in ```your-path/federated-learning-connected-speech/data/input/train``` and ```your-path/federated-learning-connected-speech/data/input/test``` (i.e., AD subjects in the ```/ad-dementia``` and healthy controls in the ```/healthy``` subfolders). The train folder should contain roughly 70% of the subjects, the test folder should contain around 30%. Ideally, the train/test folders are stratified by the label (i.e., the training and test data contain roughly the same proportion of disease groups), and roughly matched by sex/age/education (i.e., these variables do not differ significantly across train/test sets). Each subject should have a separate ```.txt``` file, the file names do not matter as long as they do not contain the terms "train"/"test"/"val". One transcript corresponds to one subject and should only contain the (automatically/manually) transcribed text. 

## External Evaluation
Here, the idea is to test a model that has been trained on several languages in a federated learning setup on a previously unseen dataset without any additional training procedure.
1. Open a terminal or use the Anaconda prompt (Windows) and activate your virtual environment (e.g., ```conda activate fl-cs```)
2. Navigate to the folder in which the ```evaluate_externally.py``` script is located (e.g., ```cd your-path/federated-learning-connected-speech/src/fl_connected_speech```)
3. Start the client (e.g., ```python client.py```). **A log file called ```external_results.log``` will be created automatically in data/output/.**

## Local Evaluation
The general idea is to investigate whether the performance of a federated learning-based model can outperform local models that are only trained on the data from each site. To measure the performance of these local models, you can run the ``client_evaluate_locally.py`` script:
1. Open a terminal or use the Anaconda prompt (Windows) and activate your virtual environment (e.g., ```conda activate fl-cs```)
2. Navigate to the folder in which the ```client_evaluate_locally.py``` script is located (e.g., ```cd your-path/federated-learning-connected-speech/src/fl_connected_speech```)
3. Start the local evaluation (e.g., ```python client_evaluate_locally.py```). **A log file called ```client_local_results.log``` will be created automatically in data/output/.**

## Starting the Client
1. Open a terminal or use the Anaconda prompt (Windows) and activate your virtual environment (e.g., ```conda activate fl-cs```)
2. Navigate to the folder in which the ```client.py``` script is located (e.g., ```cd your-path/federated-learning-connected-speech/src/fl_connected_speech```)
3. Start the client (e.g., ```python client.py```). **A log file called ```client.log``` will be created automatically in data/output/.**

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
