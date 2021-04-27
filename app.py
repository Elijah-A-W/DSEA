
from flask import Flask,render_template,request,redirect
app=Flask(__name__,static_folder='./static')


import pickle
from sklearn.externals import joblib

#importing the numpy package
import numpy as np

"""Calling the file name of the saved model"""
filename = 'knn_model.sav'

"""loading the knn saved model"""
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/',methods=["GET","POST"])
def predict():
    if request.method == "POST":
      features=[x for x in request.form.values()]

      final_features=[np.array(features)]

      prediction = loaded_model.predict(final_features)
    
      output = round(prediction[0], 4)

      return render_template('index.html',prediction=output)

    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
