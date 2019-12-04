from flask import Flask,jsonify,render_template, request, redirect,url_for,flash
import os
from functions import *

global uploadedDataframe
app = Flask(__name__)

app.config['SECRET_KEY'] = 'MYFAVORITECLASSISFE595'
app.config['FILE_UPLOADS'] = 'inputfiles'
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
        

        dfile = request.files["datafile"]
        filename = os.path.join(app.config["FILE_UPLOADS"], dfile.filename)
        dfile.save(filename)
        dsep = request.form.get("datasep")
        dheader = request.form.get("dataheader")

        uploadedDataframe = processInitialFile(filename,dheader,dsep )
        dfData = uploadedDataframe[:5]
        return render_template("uploadprocessed.html", tables=[dfData.to_html(classes='data')], titles=dfData.columns.values)
        
    else:
        return render_template("upload.html")

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