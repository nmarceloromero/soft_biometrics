# Soft Biometrics Classification

Soft biometrics define traits such as how long is someone's hair, or its t-shirt color. 
Automatically classifying those traits in images/videos could allow to improve security in public places. 
This project contains code written in Python that allows to perform soft biometrics classification in videos. 
The approach is based on the use of ransfer learning (image features extracted from the Inception network) combined to sequence models (LSTM network concretely).

## Project files

- <code>binary_evaluation.py</code>: contains functions to evaluate performance of the model.
- <code>inception_extraction.py</code>: extracts features from video frames using the pre-trained Inception network. The features are then serialized to be used afterwards.
- <code>lstm_model.py</code>: allows to build the LSTM model.
- <code>train_lstm_model.py</code>: uses the previously extracted features and trains the LSTM model to classify soft biometrics.
- <code>train_lstm_with_detection.sh</code>: a bash script to run several experiments for a certain label.

You will also find a serialized <code>.pickle</code> file inside the <code>labels_crossval</code> folder. 
This information is used by the <code>train_lstm_model.py</code> script to apply 10-fold cross-validation during training. 
This is totally optional, feel free to use your own training strategies.

## Dependencies

- tensorflow (1.3.0)
- scipy (0.19.1)
- scikit-learn (0.18.2)

## Related Paper

- [Soft_Biometrics_Classication_in_Videos_Using_Transfer_Learning_and_Bidirectional_Long_Short-Term_Memory_Network](https://sbic.org.br/lnlm/wp-content/uploads/sites/4/2020/09/vol18-no1-art4.pdf)


## Authors

- Marcelo Romero

