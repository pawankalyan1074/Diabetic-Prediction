import pickle
import numpy as np

from flask import Flask, request,render_template

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/ff',methods = ['POST', 'GET'])
def ff():
   return render_template("form.html")

@app.route('/form',methods = ['POST', 'GET'])
def predict():
   values=[]
   
   name=request.form['name']
   
   Pregnancies=request.form['Pregnancies']
   values.append(Pregnancies)
   
   Glucose=request.form['Glucose']
   values.append(Glucose)

   BloodPressure=request.form['BloodPressure']
   values.append(BloodPressure)

   SkinThickness=request.form['SkinThickness']
   values.append(SkinThickness)

   Insulin=request.form['Insulin']
   values.append(Insulin)

   BMI=request.form['BMI']
   values.append(BMI)

   DiabetesPedigreeFunction=request.form['DiabetesPedigreeFunction']
   values.append(DiabetesPedigreeFunction)

   Age=request.form['Age']
   values.append(Age)  
   
   final_values=[np.array(values)]
   
   prediction=model.predict(final_values)
   
   result=prediction
   
   if result==0:
      return render_template('result.html',name=name,Pregnancies=Pregnancies,Glucose=Glucose,BloodPressure=BloodPressure,SkinThickness=SkinThickness,Insulin=Insulin,BMI= BMI,DiabetesPedigreeFunction=DiabetesPedigreeFunction,Age=Age,rrr=0)
   else:
      return render_template('result.html',name=name,Pregnancies=Pregnancies,Glucose=Glucose,BloodPressure=BloodPressure,SkinThickness=SkinThickness,Insulin=Insulin,BMI= BMI,DiabetesPedigreeFunction=DiabetesPedigreeFunction,Age=Age,rrr=1)     

if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
