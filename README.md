# VRGameUserMotionAnalysis

This project involves the data pre-processing of Waddle VR gameplay data from the Open Game Data platform to prepare it for training a 1D Convolutional Neural Network (CNN) model. The goal is to predict whether a user will quit or continue based on their head rotations during gameplay.

To know more about the game: 

A 1D CNN model was chosen for its ability to work with sequences of data (in our case, sequences of rotation coordinates for each gameplay data entry).

The accuracy rate is roughly equal to the class distribution. This indicates that the model hasn't learnt any meaningful patterns. The low amount of data is likely the reason behind this. Further trials need to be conducted to gather more valid data and conduct more complex analyses with respect to motion, time, interactivity and user engagement rates.
