/* importing the header files */

from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__)

# change to "redis" and restart to cache again

# some time later
file=open('my_model.pkl','rb')
model=pickle.load(file)
file.close()
      


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    if request.method == 'POST':
        Status = request.form["Status"]
        Drug = request.form["Drug"]
        Age = request.form["Age"]
        Gender = request.form["Sex"]
        Ascites = request.form["Ascites"]
        Hepatomegaly = request.form["Hepatomegaly"]
        Spiders = request.form["Spiders"]
        Edema = request.form["Edema"]
        Bilirubin = request.form["Bilirubin"]
        Cholesterol = request.form["Cholesterol"]
        Albumin = request.form["Albumin"]
        Copper = request.form["Copper"]
        Alk_Phos = request.form["Alk_Phos"]
        SGOT = request.form["SGOT"]
        Tryglicerides = request.form["Tryglicerides"]
        Platelets = request.form["Platelets"]
        Prothrombin = request.form["Prothrombin"]
        
        
        data= [Status, Drug, Age, Gender, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin]
        data = np.array(data)
        data = data.astype(np.float).reshape(1,-1)
        predict = round(model.predict(data))
        
        
        
        if(predict==1):
            score='Normal!'
        elif(predict==2):
            score='Fatty Liver'
        elif(predict==3):
            score='Liver Fibrosis'
        elif(predict==4):
            score='Liver Cirrhosis'
        

    return render_template('index.html', prediction_text='The predicted liver disease is *{}* ,   Stage is : {} '.format(score,predict))

     
   


   

    #return render_template('index.html', prediction_text='The Accident Severity is *{}* ,   Score : {} / 4'.format(score,output))

"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
"""

@app.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(debug=False)
