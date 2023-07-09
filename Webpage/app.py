from flask import Flask,render_template,url_for,request,redirect,flash
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.image import imread
import random
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from IPython.display import Image

from sklearn.svm import SVC
svc=SVC(probability=True)




scalar=pickle.load(open("age_scaling.pkl","rb"))
svc=pickle.load(open("ML_model.pkl","rb"))


#UPLOAD_FOLDER = './static/uploads'
#ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, template_folder="template")
#app.secret_key='secret'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/",methods=['GET'])
def home():
	return render_template("index.html")


@app.route("/test")
def test():
	return render_template("mlsection.html")

@app.route("/scan")
def scan():
	return render_template("dlsection.html")



@app.route("/submit",methods=['POST','GET'])
def submit():
	if request.method=='POST':
		gender=request.form['gender']
		if gender=='Male':
			gender=1
		else:
			gender=0
		age=request.form['age'] 
		age_scaled=scalar.transform(np.array(age).reshape(-1,1))
		age_scaled=age_scaled[0][0]
		smoking=request.form['smoking']
		if smoking=='Yes':
			smoking=1
		else:
			smoking=0 
		yellow_fingers=request.form['yellow_fingers']
		if yellow_fingers=='Yes':
			yellow_fingers=1
		else:
			yellow_fingers=0 
		anxiety=request.form['anxiety'] 
		if anxiety=='Yes':
			anxiety=1
		else:
			anxiety=0
		peer_pressure=request.form['peer_pressure'] 
		if peer_pressure=='Yes':
			peer_pressure=1
		else:
			peer_pressure=0
		chronic_dis=request.form['chronic_dis'] 
		if chronic_dis=='Yes':
			chronic_dis=1
		else:
			chronic_dis=0
		fatigue=request.form['fatigue'] 
		if fatigue=='Yes':
			fatigue=1
		else:
			fatigue=0
		allergy=request.form['allergy'] 
		if allergy=='Yes':
			allergy=1
		else:
			allergy=0
		wheezing=request.form['wheezing']
		if wheezing=='Yes':
			wheezing=1
		else:
			wheezing=0
		alcohol=request.form['alcohol'] 
		if alcohol=='Yes':
			alcohol=1
		else:
			alcohol=0
		coughing=request.form['coughing']
		if coughing=='Yes':
			coughing=1
		else:
			coughing=0 
		short_breath=request.form['short_breath'] 
		if short_breath=='Yes':
			short_breath=1
		else:
			short_breath=0
		swallowing=request.form['swallowing'] 
		if swallowing=='Yes':
			swallowing=1
		else:
			swallowing=0
		chest_pain=request.form['chest_pain']  
		if chest_pain=='Yes':
			chest_pain=1
		else:
			chest_pain=0 
		inputs=[gender,age_scaled,smoking,yellow_fingers,anxiety,peer_pressure,chronic_dis,fatigue,allergy,wheezing,alcohol,coughing,short_breath,swallowing,chest_pain]
		inputs=np.array(inputs).reshape(1,-1)
		y=svc.predict(inputs)
		z=svc.predict_proba(inputs)
		z=round(z[0][1]*100,2)
		output=str(z)

		return render_template("mlresult.html",output=output,z=z)
		
		
		#use the below commnets to check the data of your list
        #input_set=[]
		#for i in inputs:
		#	input_set.append(" ".join(i))
		#return render_template('list.html',list=input_set)
		
        
@app.route("/base")
def base():
	return redirect(url_for('home'))	


#def allowed_file(filename):
   # return '.' in filename and \
          # filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

load_model = tensorflow.keras.models.load_model('model256x256_50epochs_resnet50.keras')
labels={0: 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',1: 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',2: 'normal',3: 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'}

@app.route('/scan',methods=['POST','GET'])
def predict():
    imagefile=request.files['imagefile']
    image_path="./static/uploads/" + imagefile.filename
    imagefile.save(image_path)
    
    image=load_img(image_path,target_size=(256,256))
    img_array = tensorflow.keras.utils.img_to_array(image)
    img_array = tensorflow.expand_dims(img_array, 0) # Create a batch
    predictions = load_model.predict(img_array)
    score = tensorflow.nn.softmax(predictions[0])
    score_max=np.argmax(score)
    get_label=labels.get(score_max)
    get_percent=int(100 * np.max(score))

    return render_template('dlsection.html',get_label=get_label,get_percent=get_percent,image="uploads/"+imagefile.filename)
    
        




if __name__=='__main__':
	app.run(debug=True)

