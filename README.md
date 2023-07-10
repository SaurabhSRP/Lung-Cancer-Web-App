# Lung-Cancer-Web-App

# Project Overview
This marks the 3rd project within my portfolio. This web app showcases three main aspects of Data science i.e.
- Data Analytics : The dashboard was created using Tableau and is available on the public forum for intreactive visualisation. The data was generated from data.gov.in website.
- Machine learning (Classification) : The preliminary_test tab in the webapp presents a form which predicts whether there is a probability of having lung cancer using ML model running in the backend. As the dataset is very small consisting of 1000 records and there is huge imbalance. Over Sampling method was implemented.
  There are two outcomes that can be seen based on the inputs.
  
  Access the code : https://github.com/SaurabhSRP/Lung-Cancer-Web-App/blob/main/ML%20model/Lung_Cancer_ML_model.ipynb
  
  --SAFE : if the probability of lung cancer is below 70% ( this value is considered for this project and may not be actual criteria)
  ![Alt text](https://github.com/SaurabhSRP/Lung-Cancer-Web-App/blob/main/Project%20snapshot/ML%20safe.png)

  --WARNING: IF the probability of lung cancer is above 70%
  ![Alt text](https://github.com/SaurabhSRP/Lung-Cancer-Web-App/blob/main/Project%20snapshot/ML%20not%20safe.png)
  
- Deep learning (image classification) : The CT-scan tab of the webapp presents the image classification using transfer learning method such as Resnet50. And you can test the model by uploading the image to see the inference given by the model. below snippet shows the output that will be generated

  Access the code : https://github.com/SaurabhSRP/Lung-Cancer-Web-App/blob/main/DL%20models/79_accuracy_Lung_Cancer_Image_Classification.ipynb

  ![Alt text](https://github.com/SaurabhSRP/Lung-Cancer-Web-App/blob/main/Project%20snapshot/DLoutput.png)

***Do check a short clip of the project***




https://github.com/SaurabhSRP/Lung-Cancer-Web-App/assets/108528607/6707ac28-cb89-48c6-9375-8cbfa01bd473

# Code and Resources Used
- ***Python Version:*** 3.8
- ***Data:*** data.gov.in , Kaggle_datalink (https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- ***Data Visualisation:*** Tableau
- ***ML algorithm:*** Scikit-learn Support Vector Machine (accuracy: 90% )
- ***Deep learning algorithm:*** keras Restnet50 transfer learning with CNN (accuracy: 79% )
- ***Frontend:*** HTML & CSS
- ***Backend:*** Flask

***Install all requirements for the web app using*** pip install -r requirements.txt


# Conclusion
- The code can be improvised even more.
- The accuracy of the Deep learning model can be improved.



 Hope you had fun :) drop your views 


