import pickle
from sklearn.externals import joblib
from api import api
import numpy as np

filename = 'knn_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

from flask import Flask,render_template,request,redirect

app=Flask(__name__,static_folder='./static')
app.register_blueprint(api,url_prefix='/api')



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