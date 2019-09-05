# Non-acted Multi-view Audio-Visual Dyadic Interactions Project
This repository contains all code generated during the development of the master thesis Non-acted Multi-view Audio-Visual Dyadic Interactions Project during the spring semester of the Master in Foundations of Data Science 2018/2019. By Aleix Casellas, Andreu Masdeu, Pablo Lázaro and Rubén Barco. 

#### Description of the project

Socially-intelligent systems have to be capable of accurately perceiving and inferring the personality and other particularities of different individuals, so as to provide a more effective, empathic, and natural tailored communication. To embody this human likeness into such systems, it is imperative to have a deeper understanding of real human-human interactions first, to computationally model both individual behavior and interpersonal interinfluence. However, there is a lack of publicly-available audiovisual databases of non-acted face-to-face dyadic interactions, which cover the richness and complexity of social communications in real life.
In this project, we collected the first of its kind non-acted audio-vidual multi-view dataset of dyadic interactions. The main goals of this dataset and associated research is to analyze human communication from a multidisciplinary perspective (i.e. technological, sociological and psychological) and to research and implement new paradigms and technologies of interpersonal behavior understanding. It is expected to move beyond automatic individual behavior detection and focus on the development of automatic approaches to study and understand the mechanisms of perception of and adaptation to verbal and non-verbal social signals in dyadic interactions, taking into account individual and dyad characteristics.

In addition to the collection of more than 80 hours of dyadic interactions including 150 participants performing cognitive tasks designed by the psychologists, this project performed a proof of concept analysis of different technical challenges included in the database: 

* Setup design, calibration and synchronization of 6 HD cameras, 2 HD egocentric cameras, 2 wrist heart rate monitors, 2 lapel microphones and 1 ambient microphone
* Multi-view joint optimization of hand and body skeleton poses for enhanced hand and body pose recovery
* Speaker audio segmentation
* Audio-visual spatio-temporal modeling of human emotions
* Multi-task face attributes analysis

The different contributions are presented below and justified in the context of their respective state-of-the-art, evaluated on proper public datasets, and finally tested as a proof of concept evaluation on the recently designed dyadic dataset.


## Pablo Lázaro Herrasti
### Multi-modal Local Non-verbal Emotion Recognition in Dyadic Scenarios and Speaker Segmentation

Study of the state-of-the-art of the emotion recognition problem using audiovisual sources. Deliver a **Emotion Recognition** system using Deep Learning techniques based on unimodal audio features, raw audio and faces, and their possible fusion. On the other hand, study of state-of-the-art of the **Speaker Segmentation** problem using audio sources and unsupervised learning techniques such as **Spectral Clustering**, experimenting with different parameters such as windows length, overlap or cluster parameters. Help and participate during the recordings of the different sessions of the **Face-to-face Dyadic Interaction Dataset** placing and collecting the setup and attending the participants. Also annotate this database labeling the utterances of the videos.

## Rubén Barco Terrones
### Multi-modal Local and Recurrent Non-verbal Emotion Recognition in Dyadic Scenarios

Study of the state-of-the-art of the **emotion recognition** problem using audiovisual sources and the different techniques that make use or not about the context, temporal information, memory blocks, etc. Deliver a emotion recognition system using Deep Learning techniques based on unimodal handcraft audio features, raw audio features and faces, and their possible fusion. Also study the influence of the temporal information to model the changes across frames in the emotion recognition problem in the unimodal and fusion experiments using **RNNs**. Help and participate during the recordings of the different sessions of the **Face-to-face Dyadic Interaction Dataset** placing and collecting the setup and attending the participants. Also annotate this database labeling the utterances of the videos.

## Aleix Casellas Comas
### Human Pose Recovery in Multi-view Dyadic Interactions

Human Pose estimation is an important field in Computer Vision that has beenextensively studied for the past few years.  It is a crucial steps towards understand-ing  people  in  image  and  videos.   Recently,  3D  pose  estimation  has  become  morepopular since its applicability in Virtual Reality, Augmented reality, etc.  There aredifferent two different approaches to obtain it: obtain first the 2D keypoints and thenlift it to 3D, or directly predit the 3D pose. The first ones are more common, and weare using this approach in the project.  We are computing the 3D human pose in amultiview setting, specifically made by 6 cameras. We first get the 2D body and handpose estimation using a state-of-the-art method (OpenPose) and then project themto 3D using multiple view geometry. To do so, we first introduce in a clear way someimportant concepts about multiple view geometry, which are key to understand allthe procedure in the 3D reconstruction.

My contribution in the project is double.  On one hand, I have been doing themulti-view joint optimization of hand and body skeleton poses for enhanced handand body pose recovery, using OpenPose and multiple view geometry. On the otherhand,  I  have  helped  and  participated  in  the  recordings  of  the  differents  sessionsof the Face-to-face Dyadic Interaction dataset.   This means placing and collectingthe setup, attending the participants and the explaining them the experiment, beingfilmed several times and annotating the facial expressions of the database.


## Andreu Masdeu Ninot
### Multitask Learning for Facial Attributes Analysis

In this thesis we explore the use of Multitask Learning for improving performance in facial attributes tasks such as gender, age and ethnicity prediction. These tasks, along with emotion recognition will be part of a new dyadic interaction dataset which was recorded during the development of this thesis. This work includes the implementation of two state of the art multitask deep learning models and the discussion of the results obtained from these methods in a preliminary dataset, as well as a first evaluation in a sample of the dyadic interaction dataset. This will serve as a baseline for a future implementation of Multitask Learning methods in the fully annotated dyadic interaction dataset.

The contribution of this master thesis within the whole project is the study of the state of the art in multitask deep learning, specially for face attributes analysis. Training and analysis of two state of the art multitask learning architectures on an external dataset and a cross-database evaluation of these trained models in the Dyadic Interaction Dataset. Comparision between multitask learning models and with single-task learning models. Help and participation during the recordings of the different sessions of the **Face-to-face Dyadic Interaction Dataset** placing and collecting the setup and attending the participants. Also collaboration in the annotation of this database labelling the
emotions of the participants.

In the MultitaskLearning folder you can find the implementation of all models in the implementations subfolder, scripts for training the models in the train subfolder and notebooks to analyze the results in the analysis subfolder.
