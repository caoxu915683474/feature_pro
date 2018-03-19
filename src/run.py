from pprint import pprint
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import pdb

def load_dict(fn):
    if isinstance(fn, list):
        sent = []
        for n in fn:
            sent += [line.strip() for line in open(n, "r").readlines() if line.strip() != ""]
        items = set(sent)
    else:
        items = set([line.strip() for line in open(fn, "r").readlines() if line.strip() != ""])
    return {item: None for item in items}

def load_dicts(*fn_list):
    match_dicts = []
    for fn in fn_list:
        if fn == "":
            match_dicts.append({})
            continue
        else:
            match_dicts.append(load_dict(fn))
    return match_dicts

def load_data(fn):
    return pd.read_csv(fn, converters={"comment":str})

def get_tfidf(corpus):
    vectorizer = CountVectorizer(min_df=5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    return word, weight

def get_weight_docu(corpus, word, weight, k=0):
    weight_corpus = {}
    for i, c in enumerate(corpus):
        seg = c.split(" ")
        weight_vec = [weight[i][word.index(s)] if s in word else 0 for s in seg]
        weight_corpus[c] = [str(x) for x in weight_vec]
    return weight_corpus

def write_dic(f, dic):
    for item in dic.items():
        f.write("%s:%s\n" %(str(item[0]), item[1]))

def run_tf_idf(data):
    word, weight = get_tfidf(data["comment"])
    tfidf_corpus = get_weight_docu(data[data["label"] == 0]["comment"], word, weight)
    return tfidf_corpus

def matcher(d, query, record):
    result = ""
    for item in d.items():
        if item[0] in query:
            record[item[0]] = record.get(item[0], 0) + 1
            if result == "" or (result != "" and len(item[0]) < len(result)):
                result = item[0]
    return "None" if result == "" else result

def report_miscoverage(df, classes):
    for c in classes:
        d = df[df["4级分类"] == c]
        print("Class %s coverage: %f" % (c, len(d[d["分类"] != "None"])/(len(d)+1)))
    df = df[""]
    print("Total coverage: %f" % (len(df[df["分类"] != "None"])/len(df)))

if __name__ == "__main__":
    match_dicts = load_dicts(
        r"dicts/fruit.dict",
        r"dicts/fruit.dict",
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        r"dicts/fruit.dict",
        r"", # Bags...
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        [r"dicts/veggie.dict", r"dicts/pigpart2.dict", r"dicts/pigpart.dict", r"dicts/pigpart3.dict"],
        [r"dicts/veggie.dict", r"dicts/rake_veggies.dict"],
        [r"dicts/pigpart2.dict", r"dicts/pigpart.dict", r"dicts/pigpart3.dict"]
    )
    data = load_data("./preprocessed.csv")

    comment = []
    labels = []
    record = {}
    c4_label_name = ["浆果", "柑橘", "调味类", "根茎类菜", "核果", "购物袋", "叶菜类", "茄果类", "结球类", "绿色", "精品蔬菜","精品"]
    doi = data[(data["label"] != 5) & (data["label"] != 9)] #[data["label"] == 2]
    for item, label in zip(doi["comment"].tolist(), doi["label"].tolist()):
        product = item.split()
        comment.append(item.replace(" ",""))
        labels.append(matcher(match_dicts[label], item.replace(" ",""), record))

    d = pd.DataFrame(columns=["商品", "4级分类", "分类"])
    d["商品"] = comment
    d["4级分类"] = [c4_label_name[int(i)] for i in doi["label"].tolist()]
    d["分类"] = labels
    d.drop_duplicates(inplace=True)
    d.to_csv("result.csv", index=False)

    report_miscoverage(d, c4_label_name)


    for c in c4_label_name:
        d2 = d[(d["4级分类"] == c) & (d["分类"] != "None")]
        if len(d2) >= 200:
            d2 = d2.sample(200)
        d2.to_csv("human_eval/%s打标签结果.csv"% c, index=False)
        
    none_items = "\n".join(d[d["分类"] == "None"]["商品"].tolist())
    print(none_items[:500])
    pprint(sorted(record.items(), key=lambda x: x[1], reverse=True))