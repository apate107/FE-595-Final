from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import os
from functions import *
from models import *


global uploadedDataframe
app = Flask(__name__)

app.config['SECRET_KEY'] = 'MYFAVORITECLASSISFE595'
app.config['FILE_UPLOADS'] = os.getcwd() + '\\inputfiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

### Home Page
@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/uploadfile", methods=["GET"])
def uploadfileGet():
    return render_template("upload.html")


@app.route("/uploadfile", methods=["POST"])
def uploadfile():
    if request.method == "POST":

        if 'datafile' not in request.files:
            flash('No file')
            return redirect(request.url)

        type = request.form.get('model')
        dfile = request.files["datafile"]
        filename = os.path.join(app.config["FILE_UPLOADS"], dfile.filename)
        dfile.save(filename)
        dsep = request.form.get("datasep")
        dheader = request.form.get("dataheader")
        uploadedDataframe = processInitialFile(filename,dheader,dsep )
        uploadedDataframe.to_pickle('inputfiles/most_recent.pkl')
        if uploadedDataframe.empty:
            flash('Data is empty')
            return redirect(request.url)
        dfData = uploadedDataframe[:5]
        results = None
        if type == 'Classification (K Nearest Neighbors)':
            results = KNN(uploadedDataframe)
        elif type == 'LinearReg(uploadedDataframe)':
            results = LinearReg(uploadedDataframe)

        return render_template("uploadprocessed.html", tables=[dfData.to_html(classes='data', index=False),
                                                               results.to_html(classes='data', index=False)],
                               titles=dfData.columns.values)

    else:
        return render_template("upload.html")


@app.route('/DecisionTree', methods = ['GET','POST'])
def decision_tree():
    if request.method == 'GET':
        uploadedDataframe = pd.read_pickle('inputfiles/most_recent.pkl')
        return render_template("decisiontree_get.html",
                               tables=[uploadedDataframe.head().to_html(classes='data')],
                               columns=uploadedDataframe.columns.tolist(),
                               titles=uploadedDataframe.columns.values,
                               depths=[i for i in range(1,6)])
    else:
        classification=request.form.get('reg')
        data = pd.read_pickle('inputfiles/most_recent.pkl')
        data.columns = [str(i) for i in data.columns]
        y = request.form.get("y")
        y = str(y)
        keeps = []

        depth = int(request.form.get("depth"))
        y_var = data.loc[:,y]
        x_vars = data.drop(y,axis=1)
        for i in range(x_vars.shape[1]):
            if type(x_vars.iloc[0,i]) in [np.dtype('float'),np.dtype('int'), np.dtype('bool')]:
                keeps.append(i)

        x_vars = x_vars.iloc[:,keeps]

        get_tree_plot(x_vars,y_var, pred_type=classification,depth=depth)

        return render_template("decisiontree_post.html",
                           titles=[x_vars.columns.values])


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    response.no_cache = True
    return response


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
