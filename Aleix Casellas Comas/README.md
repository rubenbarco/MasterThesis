# Multitask Learning For Facial Attributes Analysis

This folder contains the repository of the master thesis of **Aleix Casellas Comas** in relation with the **Non-acted Multi-view Audio-Visual Dyadic Interactions Project**.

### Description of the Master Thesis

In this thesis we explore the Human Pose recovery in multi-view dyadic interactions. The method uses computes the 2D keypoint estimation using OpenPose and then use multiple view geometry in order to project the points to 3D. This works includes the implementation of this projection to the 3D space, as well as the reprojection to 2D in order to see how good are the estimations and refine the points. 
 

### Repository Explanation

Codes for implementation and experiments are done in Jupyter notebook. All the files used are in the folder frames_experiments.

The repository is organized in the following way: 

* **frames_experiments**: In this folder one can find the files obtained using OpenPose with the estimation of the keypoints used for the notebooks. There are also the images that correspond to the frames used. They are organized in 5 subjects, each of whom has four folders inside corresponding to the frames of each camera used. Inside these last folders, there are the keypoints founded with OpenPose and the Ground Truth manually annotated.

* **3D_reconstruction_video.ipynb**: This notebook makes the 3D reconstruction of the points using the method we considered the best, displaying it into the screen. The velocity of the frames shown is not the real one.

* **Three_cameras_experiments_1.ipynb and Three_cameras_experiments_2.ipynb**: This notebook presents the 3D reconstruction of the body and hand pose for the three views setting. The first notebook is for the person 1 (left in the HC view) and the second notebook for person 2 (right in the HC view). You have to comment and uncomment, or run the appropiate cells in order to obtain the results presented in the thesis.

* **Four_cameras_experiments_1.ipynb and Four_cameras_experiments_2.ipynb**: Same as the previous notebooks, but present the case when four cameras are available for each person.





# Contact

* GitHub: [AleixCC](https://github.com/AleixCC)
* LinkedIn: [Aleix Casellas Comas](https://www.linkedin.com/in/aleix-casellas-comas-0b292b157/)
