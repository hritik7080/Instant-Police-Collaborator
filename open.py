from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import shutil


app = Flask(__name__, template_folder='./template')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=["GET", "POST"])
def login():

        conn = sqlite3.connect('./sih.db')
        data = dict()
        # conn = sqlite3.connect("E:\SIH\sih.db")
        data1 = conn.execute("SELECT * FROM car_loc")
        data2 = data1.fetchall()
        for i in data2:
            data[i] = 1

        print(data)
        data1 = conn.execute("SELECT * FROM locations")
        data2 = data1.fetchall()
        for i in data2:
            data[i] = 2
        print(data)
        data1 = conn.execute("SELECT * FROM sus_loc")
        data2 = data1.fetchall()
        for i in data2:
            data[i] = 3
        print(data)
        return render_template('profile.html', data=data)


@app.route('/log', methods=["GET", "POST"])
def log():
    user = request.form['us']
    passw = request.form['ps']
    if user == "india" and passw == "1234":
        return redirect(url_for('login'))
    else:
        return redirect(url_for('home'))



@app.route('/car', methods=["GET", "POST"])
def car():
    if request.method == 'POST':
        plate = request.form['num'].upper()
        conn = sqlite3.connect('./sih.db')
        # plate = plate.upper()
        conn.execute("Insert into plates (l_num) values (?)", [(plate)])
        conn.commit()
    return redirect(url_for('login'))


@app.route('/sus', methods=["GET", "POST"])
def sus():
    if request.method == "POST":
        path = request.form['path']
        shutil.copy(path, r'./suspects')
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run()
