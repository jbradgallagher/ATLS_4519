import multiprocessing
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np
import sys
import getopt
import nltk
import re
import math



def preprocess_text(text):
	text = re.sub('[\d+\:\d+]', '', text)
	text = re.sub('[^a-zA-Z0-9]+', ' ', text)
	text = re.sub(' +', ' ', text)
	return text.strip()


def make_trainData(inFile,useStopWords):
	outFile = ""
	all_text = open(inFile, "r", encoding='utf-8').read()
	outFile = "train_" + inFile
	outFH = open(outFile, "w", encoding='utf-8')

	if(useStopWords):
		stop_words = set(nltk.corpus.stopwords.words('english'))
		for sentence in nltk.sent_tokenize(all_text):
			aSentence = ""
			for wrd in nltk.word_tokenize(sentence):
				if wrd.lower() not in stop_words:
					aSentence = aSentence + " " + wrd.lower()
			aSentence = re.sub(r'^\s+', '', aSentence)
			outFH.write(preprocess_text(aSentence.lower())+"\n")
	else:
		for sentence in nltk.sent_tokenize(all_text):
			outFH.write(preprocess_text(sentence.lower())+"\n")

def train_word2vec(filename):
	train_filename = "train_" + filename
	data = gensim.models.word2vec.LineSentence(train_filename)
	return Word2Vec(data, vector_size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())


def getWordEmbeddings(myModel,wrdList,neighborCnt):

	embedding_clusters = []
	word_clusters = []
	for word in wrdList:
		embeddings = []
		words = []
		for similarWrd, _ in myModel.wv.most_similar(word,topn=neighborCnt):
			words.append(similarWrd)
			embeddings.append(myModel.wv[similarWrd])
		embedding_clusters.append(embeddings)
		word_clusters.append(words)

	return embedding_clusters,word_clusters

def getNeighborSimilarityTwoWords(myModel,myWord,myWord2,neighborCnt):

	words = []
	words2 = []
	scores = []
	scores2 = []

	words,scores = getNeighborSimilarity(myModel,myWord,neighborCnt)
	words2, scores2 = getNeighborSimilarity(myModel,myWord2,neighborCnt)
	
	return words,scores,words2,scores2


def getNeighborSimilarity(myModel,myWord,neighborCnt):

	words = []
	scores = []
	norms = []
	for similarWrd, score in myModel.wv.most_similar(myWord,topn=neighborCnt):
		words.append(similarWrd)
		scores.append(score)

	norms = normalizeScores(scores)
	return words,norms

def normalizeScores(scores):
	smax = -999.0
	smin = 999.0
	for score in scores:
		if(score < smin):
			smin = score
		if(score > smax):
			smax = score
	#normalize between 	0, 2*np.pi
	norm = []
	for score in scores:
		nscore = (2*np.pi)* ((score - smin)/(smax-smin))
		norm.append(nscore)
	return norm

def getAllWordEmbeddings(myModel):
	embeddings = []
	words = []

	for wrd in list(myModel.wv.key_to_index):
		embeddings.append(myModel.wv[wrd])
		words.append(wrd)

	return embeddings,words


def getWrdList(fname):
	wlist = []
	fd = open(fname, "r", encoding='utf-8')
	wlist = fd.readlines()
	fd.close()
	wlist = [wrd.rstrip() for wrd in wlist]
	return wlist

def printUsage():
	print("|(2d and 3d TSNE Plots)| python3 word2vec_viz.py -f <inFile> -p <perplexity> -n <neighbor count (for 2d)> -m <word list (for 2d)> -3 <make 3d plot>")
	print("|(circle Similarity plot)| python3 word2vec_viz.py -f <inFile> -W <word> -w <word number two (optional)> -n <neighbor count>")
	sys.exit()

def main():
	
	
	inFile = ""
	wordList = ""
	neighborCnt = 30
	prplx = 30
	useStopWords = False
	usePreTrained = False
	words = []
	scores = []
	embeddings = []
	embedding_clusters = []
	embeddings_2d = []
	embeddings_3d = []
	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:n:p:m:SP')
			for o, a in opts:
				if o == '-f':
					inFile = a
				if o == '-n':
					neighborCnt = int(a)
				if o == '-p':
					prplx = int(a)
				if o == '-m':
					wordList = a
				if o == '-P':
					usePreTrained = True

			#parse input file for training
			#train word2vec model
			if usePreTrained:
				myModel = train_word2vec_text8()
			else:
				make_trainData(inFile,useStopWords)
				myModel = train_word2vec(inFile)
			
			embeddings, words = getAllWordEmbeddings(myModel)
			embeddings = np.array(embeddings)
			tsne_3d = TSNE(perplexity=prplx, n_components=3, init='pca', n_iter=3500, random_state=12)
			embeddings_3d = tsne_3d.fit_transform(embeddings)
			cnt = 0;
			outfile = open("wordvectors.csv", "w");
			header = "cnt,x,y,x,wrd"
			outfile.write(header + "\n")
			for em,wrd in zip(embeddings_3d,words):
				outfile.write(str(cnt) + "," + str(em[0]) + "," + str(em[1]) + "," + str(em[2]) + "," + wrd + "\n")
				cnt += 1
			outfile.close()


			
			
		
					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()


