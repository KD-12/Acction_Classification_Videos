# Acction_Classification_Videos

Problem Definition:
Classifying Video based upon action performed in the video eg: swimming, cycling, applying makeup etc.

Preprocessing:
Each video is segmented into  25 frames. each frame is resized to 244, 244 using FiveCrop function of pytorch.

Extracting Features:
Features for image classification are extracted using VGG-Relu of pytorch.

Video Classification:
The features extracted from the VGG-Relu network are used for performing classification using CNN-LSTM models.

For more details on the model go trough the report file of this repo. 
Download dataset at http://crcv.ucf.edu/data/UCF101.php
