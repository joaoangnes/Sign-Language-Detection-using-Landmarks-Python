Sign Language Detection using Landmarks - Python
-
- In this project, the main objective is to create a model that identifies sign language alphabet letters, excluding letters that require specific movements to be identified.

## File Structure

The repository contains several files for training the detection model. Here's a summary of each file's functionalities:

HandTrackingModule.py

    - This file stores all the important functions for identifying hand landmarks and parameterizing the used libraries.
    - It serves as the base file for the entire project.

CollectImages.py

    - This file creates the images that will be used for training.
    - It follows the step-by-step instructions displayed on the webcam.
    - It captures 150 photos with the right hand and 150 photos with the left hand. (All parameters can be easily modified)

CreateDataSet.py

    - This file retrieves all the collected images and stores the landmarks of the 20 identified points from the mediapipe library in a 'pickle' file.

main.ipynb

    - This Jupyter notebook receives the collected data, processes it, and trains a model for identification.
    - Data normalization: ✅
    - Data balancing: ✅
    - Hyperparameter tuning: Using the RandomizedSearchCV method
    - Training Model: RandomForestClassifier
    - Validation Model: Cross Validate

    After all validations, the model is saved using pickle with the name 'model.p'.

## Utilized Libraries

- cv2
- HandTrackingModule
- Pickle
- Os

## Demonstration

Functioning Model:


## Lessons Learned

It was a great learning experience to delve into data science and complete this project.
I faced many difficulties initially in understanding how each library worked, and I admit that I still have many doubts, particularly regarding the performance of landmark detection using the mediapipe library.

In summary, this project provided valuable learning opportunities, and despite some setbacks, I managed to create a model that identifies all static sign language alphabet letters.

I have also created another repository that follows the same structure, but it uses the main hand landmark distances as input and inference data. Here's the link to the repository:
