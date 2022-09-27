from flask import Flask, render_template, request, session

import config
from query import run_query

app = Flask(__name__)
app.secret_key = "#$%#$%^%^BFGBFGBSFGNSGJTNADFHH@#%$%#T#FFWF$^F@$F#$FW"

import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

MODEL_PATH_SUMMARY = config.MODELS_PATH / 'sentence_embeddings_job_summary_clean_data_multiple_methods_250.bin'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST", "GET"])
def searchr():
    if request.method == "POST":
        query = request.form["query"]
        results = run_query(MODEL_PATH_SUMMARY, query)
        session["results"] = results
        session["query"] = query
        # return redirect(url_for("searchr"))
        return render_template("search.html", results=session["results"], query=session["query"])
    return render_template("search.html", results=session["results"], query=session["query"])


if __name__ == '__main__':
    app.run(debug=True)
