from flask import Flask, render_template, request, jsonify
from app.ml.search import query

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():
    return render_template('home_page.html')


@app.route('/search_hotels', methods=['POST'])
def search_hotels():

    r = request.form['query']
    result = query(r, ['name', 'url'], 5)
    names = result['name']
    urls = result['url']

    return render_template(
        'result_page.html',
        result=zip(names, urls),
        request=r
    )
