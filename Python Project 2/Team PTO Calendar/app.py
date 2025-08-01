from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
pto_list = []  # In-memory PTO list

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        start = request.form["start"]
        end = request.form["end"]
        pto_type = request.form["pto_type"]
        pto_list.append({
            "name": name,
            "start": start,
            "end": end,
            "pto_type": pto_type
        })
        return redirect(url_for('index'))
    return render_template("index.html", pto_list=pto_list)

if __name__ == "__main__":
    app.run(debug=True)