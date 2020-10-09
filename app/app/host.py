from flask import Flask, send_from_directory, abort
from flask import render_template, request, make_response
from zipfilehandler import ZipFileHandler
import pathlib

app = Flask("KIIT-MINOR-PROJECT")
host='127.0.0.1'
port=5000

app.config['IMAGES'] = str(pathlib.Path("static/images").resolve())
app.config['MAX_CONTENT_LENGTH'] = 3*1024*1024 #Contents Up to 3 MB allowed


@app.route("/")
def home():
    try:
        return render_template("home1.html")
    except Exception as exx:
        #return exception file
        return render_template("error.html", error=str(exx))

@app.route("/<filename>")
def template(filename):
    try:
        return render_template(filename)
    except Exception as exx:
        #return exception file
        if(str(exx) == filename):
            exx=str(exx)+" File not found"
        return render_template("error.html", error=exx)

@app.route('/favicon.ico')
def favicon():
    try:
        return send_from_directory(app.config['IMAGES'], filename='favicon.ico', mimetype='image/vnd.microsoft.icon')
    except Exception as exx:
        abort(404)

@app.route("/images/<filename>")
def images(filename):    
    try:
        return send_from_directory(app.config['IMAGES'], filename=filename)
    except FileNotFoundError:
        abort(404)

@app.route('/autoencode', methods=["POST"])
def autoencode():
    try:
        request_file = request.files['zip_file']
        if not request_file:
            return render_template("error.html", error="No File Provided")
        response = make_response(ZipFileHandler(request_file).autoEncodeDecode())
        response.headers["Content-Disposition"] = "attachment; filename=processed.zip"
        return response
    except Exception as exx:
        return render_template("error.html", error=str(exx))


if __name__ == "__main__":
    app.run(debug=True, host=host, port=port, threaded=True)
