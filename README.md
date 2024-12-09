Tilte of the Paper:Adaptive Firefly Optimization for Enhanced Vision-Based Human Gait Recognition
Requirements:
Install MATLAB latest vesrsion:MATLABR2021a/any other latest version
Run the code:
Select the main source code file:  "main_fa_modified.m" file  
Dataset:
In this work we used two openly avilable largest multiview human gait datasets
i)CASIA-B
ii)OUMVLP
Procedure:
Both the datasets consists of Gait Energy Images (GEIs).
The GEI templates undergo preprocessing, which may include normalization, noise reduction, and other techniques to enhance data quality. From the original GEI templates, derive transformed templates such as Gait Entropy Image (GEnI) and Gradient Gait Energy Image (GGEI).These transformations capture various aspects or representations of gait patterns, potentially improving feature discriminative power. 
We then apply Multiple Discriminant Analysis (MDA) to reduce dimensionality and extract features. MDA seeks to find a data projection that maximizes class separability, reducing dimensionality while preserving discriminative information. 
Subsequently, the Adaptive Firefly Optimization Algorithm (AFOA) refines the features extracted by MDA through feature selection. AFOA selects the most relevant and discriminative features from the high-dimensional space, enhancing computational efficiency and classification performance.
Finally, we used an Random forest (RF) classifier to differentiate between the Gallery set (Training) and the Probe set (Testing) for gait recognition.
Excectuion:
Intially we divided given dataset into two parts: Training set (gallery set) and Teting set (Probe set)
Where 70% of dataset used for training and 30% of dataset is used for testing.
Dataset selection: 
The CASIA-B dataset is widely recognized as a benchmark dataset for multi-view human gait recognition. This dataset comprises 124-subjects with 11-view angles (0, 18, 36,  54, 72,  90, 108, 126, 144, 162, and 180) each with a 18 phase difference.It consists of three variations, including the normal, bag, and coat conditions (size: 628MB) 
The OU-MVLP dataset is one of the largest multi-view gait datasets. The dataset comprises 10,307 subjects with 14 view angles (0, 15, 30, 45, 60, 75, 90, 180, 195, 210, 225, 240, 255, 270) and having 1.11GB size. 
Initially select any one of the datset.
Here we are providing both datasets google drive link to access data.
https://drive.google.com/drive/folders/1wjYzRTZi5_oHMdVZdPyQ3VzVHT2OmHv2?usp=drive_link
Change angles (at test set/probe set) 
for example:  ang = 90;
change directory/path
for example: list = dir('D:\Ph.D\Ph.D. WORK codes\CASIA_B_Dataset\**');
Run the code 
Finally you will get results sucessfully in the command window in terms of accuracy, sensitivity, specificity, precision, and error rate and Figure window provides confusion matrices and accuracy comparison chart.
