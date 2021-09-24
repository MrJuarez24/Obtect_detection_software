import os
from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
from shutil import rmtree
import execute_object_detection

app = Flask(__name__, template_folder = "vistas")

uploads_dir = os.path.join('static')
os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def inicio():
    return render_template('index.html')

@app.route("/upload", methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        profile = request.files['file']
        profile.save(os.path.join(uploads_dir, secure_filename(profile.filename)))

        rmtree(r"C:\Users\Javi\Desktop\UNIVERSIDAD\TFG\object_detection\src\flask_proyect\static\images")
        os.makedirs(r"C:\Users\Javi\Desktop\UNIVERSIDAD\TFG\object_detection\src\flask_proyect\static\images")

        # ("a","b")
        fotos, alarmas = execute_object_detection.main(os.path.join(uploads_dir, secure_filename(profile.filename)), r"C:\Users\Javi\Desktop\UNIVERSIDAD\TFG\output\annotation.json", r"C:\Users\Javi\Desktop\UNIVERSIDAD\TFG\object_detection\src\flask_proyect\static\images")


    return render_template('mostrar_archivo.html', imagenes = fotos, alarms = alarmas)

if __name__ == '__main__':
    app.run(debug=True)

