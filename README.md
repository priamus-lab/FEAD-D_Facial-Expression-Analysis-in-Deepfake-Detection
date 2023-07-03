# FEAD-D: Facial Expression Analysis in Deepfake Detection
As the development of deep learning (DL) techniques has progressed, the creation of convincing synthetic media, known as deepfakes, has become increasingly easy, raising significant concern about the use of these videos to spread false information and potentially manipulate public opinion. In recent years, deep neural networks, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), have been used for deepfake detection systems, exploiting the inconsistencies and the artifacts introduced by generation algorithms. Taking into account the main limitation of fake videos to realistically reproduce the natural human emotion patterns, we present FEAD-D, a publicly available tool for deepfake detection performing facial expression analysis. Our system exploits data from the DeepFake Detection Challenge (DFDC) and consists of a model based on bidirectional Long Short-Term Memory (BiLSTM) capable of detecting a fake video in about two minutes with an overall accuracy of 84.29% on the test set (i.e. comparable with the current state-of-the-art, while consisting of fewer parameters), showing that emotional analysis can be used as a robust and reliable method for deepfake detection.


## Running the code
* Download the weights of the models at this link: https://drive.google.com/drive/folders/1jeoqIl6SH5eRZooVesM2gqtBuTxOJbiU?usp=sharing
* Open the FEADD-Classification.ipynb on Google Colab

## Alternative workflow
We also implemented an alternative workflow by considering the face detector used in the [Selim Seferbekov](https://github.com/selimsef/dfdc_deepfake_challenge) and in [Davide Coccomini](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection) implementations.  
In the Alternative workflow folder:
* detect_faces.py is used to compute the coordinated of the box including the human face
* extract_crops_us.py is used to extract the human face
* extracFeatures.py is used to extract the textural and emotional features as proposed in our methodology
* classification.py is used to provide the prediction for each test video.


## Authors
Michela Gravina, Antonio Galli, Geremia De Micco, Stefano Marrone, Giuseppe Fiameni, Carlo Sansone
