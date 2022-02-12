from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   exp= float(request.values['Stock'])
   exp=np.reshape(exp,(-1,1))
   output=model.predict(exp)
   output=output.item()
   output=round(output)
   return render_template ('result.html',prediction_text="The StockPrice is.{}".format(output))
if __name__=='__main__':
    app.run(port=8000)
