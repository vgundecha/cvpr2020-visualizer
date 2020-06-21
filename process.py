import os
import urllib.request
import re
import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

base_path = "http://openaccess.thecvf.com/content_CVPR_2020/papers/"
    
# download papers
with open("urls.txt", 'r') as f:
    urls = [url for url in f.read().split("\n")]
    
with open("titles.txt", 'r') as f:
    titles = [title for title in f.read().split("\n")]
 
urls.remove("")
titles.remove("")

folder = "papers"
if os.path.exists(folder):
    print("Found papers folder, skipping download")
else:
    print("Downloading %d papers" % (len(urls)))
    os.mkdir(folder)
    for url in tqdm.tqdm(urls):
        urllib.request.urlretrieve(base_path + url, filename=os.path.join(folder, url))

    # convert to pdf
    print("Converting pdf to txt")
    for url in tqdm.tqdm(urls):
        os.system("cd %s && pdftotext %s" % (folder, url))
    
    
# train
paths = [os.path.join(folder, path[:-3] + 'txt') for path in urls]

def train_x(paths):
  for p in paths:
    with open(p, 'rb') as f:
      txt = f.read()
    yield txt
    
max_features = 1000
v = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features = max_features, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=1)

X = train_x(paths)
print("Training TF-IDF")
v.fit(X)

# save feature matrix
X = train_x(paths)
X = v.transform(X)
print(X.shape)

np.savetxt("X_new.tsv", X.toarray(), delimiter='\t')

# for tensorboard
with open("labels.tsv", 'w') as f:
    f.write("Title\tURL\n")
    for title, url in zip(titles, urls):
        f.write("%s\t%s\n" % (title, base_path + url))
