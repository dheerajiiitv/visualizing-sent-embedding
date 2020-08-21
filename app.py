from flask import Flask, render_template, request
from  visualize_USE import get_important_words, get_visual_representation, scores
import json
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/get_relevance', methods=['post'])
def get_relevance():
    print("Finding Word Imrportance")
    user_text  = request.json['user_text']
    #TODO: Preprocessing in input text
    print("User text", user_text)

    word_importance = get_important_words(user_text, user_text)
    word_score_tuple = list(map(lambda x, y: (x, y), user_text.split(), word_importance))
    word_score_tuple.sort(key=lambda x: -x[1])
    return json.dumps({'text':get_visual_representation(word_importance, user_text),
                       'word_importance':"\n".join([" -> ".join(map(str,i)) for i in word_score_tuple][:6])})



if __name__ == '__main__':
    app.run()
