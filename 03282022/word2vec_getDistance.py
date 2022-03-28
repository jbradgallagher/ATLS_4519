import multiprocessing
import gensim
from gensim.models import Word2Vec
import gensim.downloader as api
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
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

def train_word2vec_text8():
	corpus = api.load("text8")
	return Word2Vec(corpus, vector_size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())

def getPreTrainedModel():
	model = api.load("fasttext-wiki-news-subwords-300")
	return model

def getWordPairEmbeddingsPT(myModel,wrdPairs):
	embeddings_wordOne = []
	embeddings_wordTwo = []
	foundWrdPairs = {}

	for key in wrdPairs:
		#make sure both words are in the word vector model
		if key in list(myModel.key_to_index) and wrdPairs[key] in list(myModel.key_to_index):
			embeddings_wordOne.append(myModel[key])
			embeddings_wordTwo.append(myModel[wrdPairs[key]])
			foundWrdPairs[key] = wrdPairs[key]
		else:
			print("Words not found: ",key,wrdPairs[key])

	print("Loaded Embeddings: ",len(embeddings_wordOne),len(embeddings_wordTwo))
	return embeddings_wordOne,embeddings_wordTwo,foundWrdPairs

def getWordPairEmbeddings(myModel,wrdPairs):
	embeddings_wordOne = []
	embeddings_wordTwo = []
	foundWrdPairs = {}

	for key in wrdPairs:
		#make sure both words are in the word vector model
		if key in list(myModel.wv.key_to_index) and wrdPairs[key] in list(myModel.wv.key_to_index):
			embeddings_wordOne.append(myModel.wv[key])
			embeddings_wordTwo.append(myModel.wv[wrdPairs[key]])
			foundWrdPairs[key] = wrdPairs[key]
		else:
			print("Words not found: ",key,wrdPairs[key])

	print("Loaded Embeddings: ",len(embeddings_wordOne),len(embeddings_wordTwo))
	return embeddings_wordOne,embeddings_wordTwo,foundWrdPairs


	
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

def getWrdPairs(fname):
	wlist = []
	wpairs = {}
	fd = open(fname, "r", encoding='utf-8')
	wlist = fd.readlines()
	fd.close()
	for line in wlist:
		wrd1,wrd2 = line.rstrip().split(',')
		wpairs[wrd1] = wrd2
	return wpairs

def printUsage():
	print("python3 word2vec_getDistance.py -f <inFile (if training own corpus)> -T -m <wrdListFile>" + "\n")
	print("python3 word2vec_getDistance.py -m <wrdListFile> -P (For training with pre-trained word vectors see models.gensim" + "\n")
	print("python3 word2vec_getDistance.py -m <wrdListFile> -T -8 (for training with text8 corpus)" + "\n")
	sys.exit()

def main():
	
	
	inFile = ""
	wordList = ""
	neighborCnt = 30
	prplx = 30
	useStopWords = True
	usePreTrained = False
	useTrainCorpus = False
	useTrainText8 = False
	words = []
	scores = []
	embeddings = []
	embedding_clusters = []
	embeddings_2d = []
	embeddings_3d = []

	embeddings_one = []
	embeddings_two = []
	embeddings_3d_one = []
	embeddings_3d_two = []
	
	myWrdPairs = {}
	myFoundWrdPairs = {}

	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:n:p:m:PT8')
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
				if o == '-T':
					useTrainCorpus = True
				if o == '-8':
					useTrainText8 = True

			#parse input file for training
			#train word2vec model
			if usePreTrained:
				#get a pretrained model (see models.gensim in this directory)
				#must change the model name in the getPreTrainedModel call
				myModel = getPreTrainedModel()
				myWrdPairs = getWrdPairs(wordList)
				embeddings_one, embeddings_two, myFoundWrdPairs = getWordPairEmbeddingsPT(myModel,myWrdPairs)
			else:
				#train your own corpus, filename is inFile for -f option
				if(useTrainCorpus and not useTrainText8):
					make_trainData(inFile,useStopWords)
					myModel = train_word2vec(inFile)
				else:
					#train the "text8" corpus that gensim provides
					if(useTrainCorpus and useTrainText8):
						myModel = train_word2vec_text8();

				myWrdPairs = getWrdPairs(wordList)
				embeddings_one, embeddings_two, myFoundWrdPairs = getWordPairEmbeddings(myModel,myWrdPairs)

			tsne_3d = TSNE(perplexity=prplx, n_components=3, init='pca', n_iter=3500, random_state=12)
			embeddings_3d_one = tsne_3d.fit_transform(embeddings_one)
			embeddings_3d_two = tsne_3d.fit_transform(embeddings_two)


			cnt = 0;
			outfile = open("wordDistance.csv", "w");
			header = "cnt,wrdOne,wrdTwo,x1,y1,z1,x2,y2,z2,distance"
			outfile.write(header + "\n")
			for emOne,emTwo,wrdKey in zip(embeddings_3d_one,embeddings_3d_two,myFoundWrdPairs):
				xDist = math.sqrt((emTwo[0] - emOne[0]) * (emTwo[0] - emOne[0]))
				yDist = math.sqrt((emTwo[1] - emOne[1]) * (emTwo[1] - emOne[1]))
				zDist = math.sqrt((emTwo[2] - emOne[2]) * (emTwo[2] - emOne[2]))
				dist = math.sqrt(xDist+yDist+zDist);
				outfile.write(str(cnt) + "," + wrdKey + "," + myWrdPairs[wrdKey] + "," + str(emOne[0]) + "," \
					+ str(emOne[1]) + "," + str(emOne[2]) + "," + str(emTwo[0]) + "," + str(emTwo[1]) + "," + \
					str(emTwo[2]) + "," + str(dist) + "\n")
				cnt += 1
			outfile.close()

					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()


