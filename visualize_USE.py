from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import toyplot as tps
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# Finding word importance.
def scores(text1, text2):
    vectors = embed([text1, text2])
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]



def get_important_words(original_text, reference_text):
    original_score = scores(original_text, reference_text)
    text_ls = word_tokenize(original_text)
    len_text = len(text_ls)
    leave_1_texts = [text_ls[:ii] + [''] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    new_similarity_scores = list(
        [original_score  - scores(TreebankWordDetokenizer().detokenize(i), reference_text) for i in leave_1_texts])
    return new_similarity_scores


def get_color(score):
    if score > 0:
        return tps.color.brewer.map('Greens').css(1 - score, domain_min=0, domain_max=1)
    elif score < 0:
        return tps.color.brewer.map('Reds').css(1 + score, domain_min=0, domain_max=1)
    else:
        return 'rgba(0,0,0)'


def get_visual_representation(scores, user_text):
    modified_user_text = []
    tokenized_user_text = word_tokenize(user_text)
    for score, word in zip(scores, tokenized_user_text):
        color = get_color(score)
        new_word = '<span style="background-color:' + color + '">' + word + '</span>'
        modified_user_text.append(new_word)

    return TreebankWordDetokenizer().detokenize(modified_user_text)