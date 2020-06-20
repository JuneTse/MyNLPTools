#coding:utf-8
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import logging

logger=logging.getLogger('word2vec')
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
logging.root.setLevel(level=logging.INFO)

def train_skipgram(corpus_path,emb_dim,model_path,iter=10):
    '''
    sg=1: 使用skip-gram
    '''
    model=Word2Vec(sentences=LineSentence(source=corpus_path),size=emb_dim,min_count=1,window=10,iter=iter,sg=1,negative=50)
    model.wv.save_word2vec_format(model_path,binary=False)
def train_cbow(corpus_path,emb_dim,model_path,iter=10):
    '''
    sg=1: 使用skip-gram
    '''
    model=Word2Vec(sentences=LineSentence(source=corpus_path),size=emb_dim,min_count=1,window=10,iter=iter,sg=0,negative=50)
    model.wv.save_word2vec_format(model_path,binary=False)

def load_word2vec(model_path):
    model=KeyedVectors.load_word2vec_format(model_path,binary=False)
    return model

def binary2text(binary_model_path,save_path):
    model=KeyedVectors.load_word2vec_format(binary_model_path,binary=True)
    model.wv.save_word2vec_format(save_path,binary=False)


if __name__=="__main__":
    # word="广东捷信融资担保有限公司"
    # model=load_word2vec(model_path="chars2vec.bin")
    word="hours"
    # model=load_word2vec(model_path="vocab_bert.bin")
    model=load_word2vec(model_path="embeddings_wn18rr_glove.bin")
    sims = model.most_similar(word, topn=50)
    print(sims)
    # model=load_word2vec(model_path="word2vec_glove_128.bin")
    # sims=model.most_similar(word,topn=50)
    # print(sims)
    # binary2text(binary_model_path="chars2vec_model.bin",save_path="chars2vec.bin")
