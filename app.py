from flask import Flask,jsonify,render_template, request, redirect,url_for
import os


app = Flask(__name__)

app.config['SECRET_KEY'] = 'MYFAVORITECLASSISFE595'
app.config['FILE_UPLOADS'] = 'inputfiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

### Home Page
@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/uploadfile", methods=["GET", "POST"])
def uploadfile():
    if request.method == "POST":

        if request.files:

            dfile = request.files["datafile"]

            dfile.save(os.path.join(app.config["FILE_UPLOADS"], dfile.filename))

            
            return redirect('/')
    return render_template("upload.html")

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)