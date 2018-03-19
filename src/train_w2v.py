import pandas as pd
import sys
from gensim.models import Word2Vec

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage : %s <data.csv> <model_output.txt>" % sys.argv[0])
        sys.exit(-1)

    print("Loading and splitting sentences...")
    df = pd.read_csv(sys.argv[1], converters={"comment": str})
    sents = [line.split() for line in df["comment"].tolist()[:]]
    print(len(sents))
    print(sents[:20])

    print("Continue to w2v training...")
    model = Word2Vec(sents, size=200, min_count=5, workers=10, sg=1)
    model.wv.save_word2vec_format(sys.argv[2], binary=False)

