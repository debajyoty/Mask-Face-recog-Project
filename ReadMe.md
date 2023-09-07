### Enhancing Security with Optimized Masked Face Recognition and Mask Detection

## About the Project

“It is a dual purpose model: Optimized Masked FaceRecognition and Mask Detection for Half-CoveredFaces” is a solution to the problem of recognition of faces wearing masks, especially during the pandemic by COVID-19. This file consists of detailed step by step procedure to run our model as well as we will be stating all pre-requisits which are needed to run the code. 

### Built With 

This section consists of frameworks that helped us build the project. They are :
	
* Python 3.8 
* Anaconda Navigator
* Spyder 4.1.5

## Getting started
Now we get into setting up our project in your local system. To get a local copy and running, follow these 

### Prerequisites

We start by first installation of [Anaconda Navigator](https://www.anaconda.com/products/individual).
**Please Note** Do check the box during installation which reads "Add to path". 
This is important because it helps in easier installation of packages written in the following sections.

Once Anaconda Navigator is installed, set up a new environment and install Spyder 4.1.5. 

### Installing packages

Once the Anaconda is set up, open up the windows terminal and set the environment to the project enironment. 
Our Environment name is Project_faceRec so to set the environment we use 
'''sh
activate Project_faceRec
'''

This will activate the Environment in which the project will run in. Now installation the following modules from this terminal itself needs to be done in order to collect all dependencies of the code. 

1. Installing Open CV 4.4.0 
'''sh
conda install -c conda-forge opencv
'''
2. Installing Json5 0.9.5  
'''sh
Pip install jsonlib
'''
3. Installing Imutils 0.5.3 
'''sh
conda install -c conda-forge imutils
'''
4. Installing Shutil 1.7.0
'''sh
Pip install pytest-shutil
'''

Once these dependencies are downloaded from the terminal, dependencies from the Anaconda terminal needs to be installed in our working environment. 

* Keras 2.4.3
* Keras-preprocessing 1.1.0 
* Tensorflow 2.1.0
* Argparse

## System Requirements 

This section consists of the the system requirements to execute the code are the specifications of a consumer level PC. We will still be specifying the system of our use as well as the minimum requirements for the same.

* Intel i5 8th Gen Base 1.80 GHz (base) 3.9GHz (Max) {Any processor with base speed 1.80GHz and max CPU speed of 3.5GHz will be suitable for running the code)
* RAM 8.00 GB {Min RAM required 4.00 GB)
* 64-bit operating system to ensure no hassle for python 3.8 and Anaconda Navigator.

## Usage 

This section encloses every detail of how to run the project in your local system once all the above mentioned dependencies are successfully installed in the working environment. Before explaining how to use the source code, we will decribe the files and folders present in this repository.

* Maskdetection-master :  This folder consists of the mask detection model which executes with the final model of face recognition too. **Please Note that this file need not be executed.** There are three python files for creating the mask detector, training the mask detector and then implementation of the trained mask deteector.
* Dataset : There exists a zip file named dataset. We need to extract it into the folder named **'Recog_Train'.** If one wants to add their ID to the databes then they must make a folder with 3-4 pictures of themselves in the 'Recog_Train' folder. 
* facenet_keras.h5 : Supporting file for helping model detect faces. 
* classifier.h5 : Supporting file for helping the model classify IDs for recognition purpose
* recog_data.npz & portal_data.npz : It is the zipped archive of files named after the variables they contain and are needed while running the code.

Now that all files and their usage has been disccussed, we move into important sections of source code that one needs to look into. 



## Code Segments 

Before moving into the source code itself, one needs to set the current working directory where all files are exctracted too. *It is recomended to not use deep paths as they tend to creat problems for the compiler to follow.*  We then open the Face_recog.py file in spider and look into the following changes.


**NOTE** The Face_recog.py implements the already trained face recognition of masked face. There is no extra source code for testing and training the model separately. 

If all paths, dependencies and folders are in their correct places, we now simply run the source code. It would open up the camera of your system (please ensure that there is a camera device connected to the system) and display the desired output i.e. it will recognize one's face and also detect whether one is wearing mask or not. 


**NOTE** The file contains a folder called testing.py. It is the source code just in case one wants to see the results of the model. The testing.py file imports the .npz files created earlier and then performs testing on them.

**To Add unknown candidate** Goto Recog_Train, Then add a folder with images of that person whose identity we want to add on the database. Rename the folder as the new person's name.

## Conclusion 

This project is craeted in order to provide a solution to the problem of masked face recognition. Any changes or enhance to the project is welcomed to be contributed with proper citation. 
