# Human-Activity-Recognition(HAR) Video Classification Using CNN-RNN Architecture

Human Activity Recognition (HAR) using video data involves recognizing and classifying actions performed by humans in videos. A common approach for this task is to combine Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). This hybrid architecture leverages the CNN to extract spatial features from individual video frames, while the RNN (typically LSTM or GRU)here i used LSTM handles the temporal sequence of these frames.
Here's how it typically works:

1. CNN for Feature Extraction:
CNNs are well-suited for processing visual data and can extract spatial features (such as movement patterns, object detection, etc.) from video frames.

A popular approach is to pass individual frames (or batches of frames) from a video through a pre-trained CNN (such as ResNet, VGG, or MobileNet).
The CNN processes the spatial information in each frame and converts it into a high-dimensional feature representation.

2.RNN for Sequence Modeling:
After the CNN extracts features from the frames, the features are passed into the RNN. The RNN learns how these features evolve over time and captures long-term dependencies between them.

LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Units) are commonly used because they are effective at handling long sequences and mitigating the vanishing gradient problem.
