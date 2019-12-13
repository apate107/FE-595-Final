
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, g
import os
import numpy as np
from functions import *
import datetime as dt


app = Flask(__name__)
app.app_context()
app.config['SECRET_KEY'] = 'MYFAVORITECLASSISFE595'
app.config['FILE_UPLOADS'] = os.getcwd() + '/inputfiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# Home Page

@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/KNN/", methods=["GET"])
@app.route("/DecisionTree/", methods=["GET"])
@app.route("/uploadfile", methods=["GET"])
def uploadfileGet():
    return render_template("upload.html")


@app.route("/uploadfile", methods=["POST"])
def uploadfile():
    
    if request.method == "POST":

        dfile = request.files["datafile"]
        dfile.seek(0, os.SEEK_END)
        if dfile.tell() == 0:
            flash('No file was detected, please select a file')
            return redirect( url_for('uploadfileGet'))

        dfile.seek(0)
        dsep = request.form.get("datasep")
        dheader = request.form.get("dataheader")
        
        uploadedDataframe = processInitialFile(dfile, dheader, dsep)
        input_fileName = 'input_fileName' + dt.datetime.now().strftime("%m%d%y%h%m%S%f")    
        input_file = os.path.join(app.config['FILE_UPLOADS'],input_fileName)
        
        uploadedDataframe.to_pickle(input_file)
        
        if uploadedDataframe.empty:
            flash('Data is empty')
            return redirect(request.url)
        dfData = uploadedDataframe[:5]

        return render_template("uploadprocessed.html", tables=[dfData.to_html(classes='data', index=False)],
                               titles=dfData.columns.values, filename=input_fileName)

    else:
        return render_template("upload.html")


@app.route('/KNN/<filename>', methods=['GET', 'POST'])
def knn(filename):
    input_file = os.path.join(app.config['FILE_UPLOADS'],filename)
    if request.method == 'GET':
        uploadedDataframe = pd.read_pickle(input_file)
        return render_template("knn_get.html",
                               tables=[uploadedDataframe.head().to_html(classes='data', index=False)],
                               columns=uploadedDataframe.columns.tolist(),
                               titles=uploadedDataframe.columns.values,
                               k_values=[i for i in range(1, len(uploadedDataframe)+1)],
                               filename=filename)
    else:
        data = pd.read_pickle(input_file)
        data.columns = [str(i) for i in data.columns]
        y = request.form.get("y")
        y = str(y)
        keeps = []

        k = int(request.form.get("k"))
        y_var = data.loc[:, y]
        x_vars = data.drop(y, axis=1)
        for i in range(x_vars.shape[1]):
            if type(x_vars.iloc[0, i]) in [np.dtype('float'), np.dtype('int'), np.dtype('bool')]:
                keeps.append(i)
        x_vars = x_vars.iloc[:, keeps]
        res = get_knn_plot(x_vars, y_var, k=k)

        return render_template("knn_post.html", k=k, acc=str(round(res['accuracy']*100, 2))+'%',
                               filename=filename)


@app.route('/DecisionTree/<filename>', methods=['GET', 'POST'])
def decision_tree(filename):
    input_file = os.path.join(app.config['FILE_UPLOADS'],filename)
    
    if request.method == 'GET':
        uploadedDataframe = pd.read_pickle(input_file)
        return render_template("decisiontree_get.html",
                               tables=[uploadedDataframe.head().to_html(classes='data')],
                               columns=uploadedDataframe.columns.tolist(),
                               titles=uploadedDataframe.columns.values,
                               depths=[i for i in range(1,6)],
                               filename=filename)
    else:
        classification=request.form.get('reg')
        data = pd.read_pickle(input_file)
        data.columns = [str(i) for i in data.columns]
        y = request.form.get("y")
        y = str(y)
        keeps = []

        depth = int(request.form.get("depth"))
        y_var = data.loc[:, y]
        x_vars = data.drop(y, axis=1)
        for i in range(x_vars.shape[1]):
            if type(x_vars.iloc[0, i]) in [np.dtype('float'), np.dtype('int'), np.dtype('bool')]:
                keeps.append(i)

        x_vars = x_vars.iloc[:, keeps]

        outs = get_tree_plot(x_vars, y_var, pred_type=classification, depth=depth)
        if classification == 'Classification':
            metrics1 = 'In-Sample Accuracy: '
            metrics2 = 'OOS Accuracy: '
        else:
            metrics1='In-Sample MSE: '
            metrics2 = 'OOS MSE: '

        m1=outs[0]
        m2 = outs[1]


        return render_template("decisiontree_post.html", titles=[x_vars.columns.values],
                               filename=filename, metrics1 = metrics1,metrics2=metrics2,
                               m1=m1,m2=m2)



@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    response.no_cache = True
    return response


if __name__ == '__main__':
    

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

