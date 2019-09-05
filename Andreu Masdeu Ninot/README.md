# Multitask Learning For Facial Attributes Analysis

This folder contains the repository of the master thesis of **Andreu Masdeu Ninot** in relation with the **Non-acted Multi-view Audio-Visual Dyadic Interactions Project**.

### Description of the Master Thesis

In this thesis we explore the use of Multitask Learning for improving performance in facial attributes tasks such as gender, age and ethnicity prediction. These tasks, along with emotion recognition will be part of a new dyadic interaction dataset which was recorded during the development of this thesis. This work includes the implementation of two state of the art multitask deep learning models and the discussion of the results obtained from these methods in a preliminary dataset, as well as a first evaluation in a sample of the dyadic interaction dataset. This will serve as a baseline for a future implementation of Multitask Learning methods in the fully annotated dyadic interaction dataset.
 

### Repository Explanation

Codes for implementation and training of models are Python scripts. For analysing results jupyter notebook were created instead.

The repository is organized in the following way, in three folders: 

* **implementations**: Here you can find the implementations for building the Hard Parameter Sharing Models and for building the Cross-stitch Networks. To build a Cross-stitch Network a custom Keras layer for the cross-stitch operation is necessary and is implemented in a class called ProportionalAddition inside the leaky_unit.py file. This file is named after another multitask learning unit named leaky_unit who was finally not included in the thesis.

* **train**: Here are all scripts coded to train all possible configurations of the MTL models. The Baseline Model is trained in the Cross-stitch networks scripts. There are 16 files, since each file contains a different grouping of the tasks (4 different groupings: 3 pairs and the triplet) and also each file is doubled for the backwards and normal approach.

* **analysis**: Here there are different notebooks for analysing the results obtained during training. 'Create UTK Results Table.ipynb' creates the tables containing all performance metrics in each task and for each configuration. 'Create Dyadic Results Table.ipynb' does the same but for the dyadic dataset. CreateDataset.ipynb generates the dyadic dataset from the raw videos and ClassActivationMap.ipynb allows to visualize CAM's from images of the UTKFace dataset. Finally 'Analyse Results Table.ipynb' and 'Analyse Results Table-Dyadic.ipynb' generate the figures shown in the Results section from the results table.



# Contact

* GitHub: [andreu15](https://github.com/andreu15)
* LinkedIn: [Andreu Masdeu Ninot](https://es.linkedin.com/in/andreu-masdeu-ninot-23139714a)
