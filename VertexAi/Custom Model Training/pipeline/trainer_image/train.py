
import numpy as np
import pandas as pd
from os import system
import fire
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load # used for saving and loading sklearn objects
from scipy.sparse import save_npz, load_npz, csr_matrix # used for saving and loading sparse matrices

def train_model(training_dataset_file, validation_dataset_file, gcs_model_path):
    
    imdb_train = pd.read_csv(training_dataset_file)
    imdb_test = pd.read_csv(validation_dataset_file)
    
    if not os.path.exists('data_preprocessors'):
        system("mkdir 'data_preprocessors'")
    if not os.path.exists('vectorized_data'):
        system("mkdir 'vectorized_data'")
    if not os.path.exists('model'):
        system("mkdir 'model'")

    #preprocessing

    # Bigram Counts

    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    bigram_vectorizer.fit(imdb_train['text'].values)
    
    dump(bigram_vectorizer, 'data_preprocessors/bigram_vectorizer.joblib')
    
    # bigram_vectorizer = load('data_preprocessors/bigram_vectorizer.joblib')
    
    X_train_bigram = bigram_vectorizer.transform(imdb_train['text'].values)
    
    save_npz('vectorized_data/X_train_bigram.npz', X_train_bigram)
    
    # X_train_bigram = load_npz('vectorized_data/X_train_bigram.npz')
    
    # Bigram Tf-Idf

    bigram_tf_idf_transformer = TfidfTransformer()
    bigram_tf_idf_transformer.fit(X_train_bigram)

    dump(bigram_tf_idf_transformer, 'data_preprocessors/bigram_tf_idf_transformer.joblib')

    # bigram_tf_idf_transformer = load('data_preprocessors/bigram_tf_idf_transformer.joblib')

    X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

    save_npz('vectorized_data/X_train_bigram_tf_idf.npz', X_train_bigram_tf_idf)
    
    # X_train_bigram_tf_idf = load_npz('vectorized_data/X_train_bigram_tf_idf.npz')

    y_train = imdb_train['label'].values

    def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
        #splitting data
        X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
        )

        clf = SGDClassifier()
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        valid_score = clf.score(X_valid, y_valid)
        print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')
        dump(clf, 'model/model.joblib.pkl')
        subprocess.check_call(['gsutil', 'cp', 'model/model.joblib.pkl', gcs_model_path],
                        stderr=sys.stdout)
        print('Saved model in: {}'.format(gcs_model_path))

    train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
    
if __name__ == '__main__':
  fire.Fire(train_model)
