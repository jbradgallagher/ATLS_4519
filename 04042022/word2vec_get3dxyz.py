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



def make_Concordance(inFile,useStopWords):
	concord = {}
	stop_words = []
	all_text = open(inFile, "r", encoding='utf-8').read()

	maxFreq = -9999999;
	if(useStopWords):
		stop_words = set(nltk.corpus.stopwords.words('english'))
		for sentence in nltk.sent_tokenize(all_text):
			for wrd in nltk.word_tokenize(sentence):
				myWrdPos = nltk.pos_tag(nltk.word_tokenize(wrd))
				if wrd.lower() not in stop_words and (myWrdPos[0][1] == 'NN' or myWrdPos[0][1] == 'JJ' or re.search("^V", myWrdPos[0][1])):
					if wrd.lower() not in concord:
						concord[preprocess_text(wrd.lower())] = 1
					else:
						concord[preprocess_text(wrd.lower())] += 1
			
	return concord

def getMaxCountWords(concord,maxFreq):
	maxCountWords = []
	for wrd in concord:
		if(concord[wrd] >= maxFreq):
			maxCountWords.append(wrd)
	return maxCountWords


def getPreTrainedModel(preTrainedWordVectors):
	model = api.load(preTrainedWordVectors)
	return model
	
def getWordEmbeddings(myModel,wrdList):
	embeddings = []
	foundWordList = []
	for word in wrdList:
		if word in myModel:
			embeddings.append(myModel[word])
			foundWordList.append(word)
		else:
			print("WORD not found",word)
	return embeddings, foundWordList
		
def getWrdList(fname):
	wlist = []
	fd = open(fname, "r", encoding='utf-8')
	wlist = fd.readlines()
	fd.close()
	wlist = [wrd.rstrip() for wrd in wlist]
	return wlist

def printUsage():
	print("python3 word2vec_get3dxyz.py -f <inFile> -P <pretrained word vectors> -8 <text8 corpus> -m <wrdListFile> -p <perplexity>" + "\n")
	sys.exit()

def main():
	
	
	inFile = ""
	wordListFile = ""
	preTrainedWordVectors = ""
	neighborCnt = 30
	prplx = 30
	useStopWords = True
	useCorpus = False
	useWordList = False
	embeddings = []
	embeddings_3d = []
	
	myWrdList = []
	foundWordList = []
	myConcord = {}
	maxFreq = 0
	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:p:m:P:M:')
			for o, a in opts:
				if o == '-f':
					inFile = a
					useCorpus = True
				if o == '-m':
					wordListFile = a
					useWordList = True
				if o == '-M':
					maxFreq = int(a)
				if o == '-p':
					prplx = int(a)
				if o == '-P':
					preTrainedWordVectors = a
				
			if(useWordList):
				myWrdList = getWrdList(wordListFile)
			if(useCorpus):
				myConcord = make_Concordance(inFile,useStopWords)
				myWrdList = getMaxCountWords(myConcord,maxFreq)
				print(myWrdList)
			#get a pretrained model (see models.gensim in this directory)
			myModel = getPreTrainedModel(preTrainedWordVectors)
			embeddings,foundWordList = getWordEmbeddings(myModel,myWrdList)
			
		
			tsne_3d = TSNE(perplexity=prplx, n_components=3, init='pca', n_iter=3500, random_state=12)
			embeddings_3d = tsne_3d.fit_transform(embeddings)
		

			cnt = 0;
			outfile = open("myWordsXYZ.csv", "w")
			outfile2 = open("myMesh.txt", "w")
			header = "cnt,wrd,x,y,z"
			outfile.write(header + "\n")
			for em3d,wrd in zip(embeddings_3d,foundWordList): 
				outfile.write(str(cnt) + "," + wrd + "," + str(em3d[0]) + "," + str(em3d[1]) + "," + str(em3d[2]) + "\n")
				outfile2.write(str(em3d[0]) + "," + str(em3d[1]) + "," + str(em3d[2]) + "\n")
				cnt += 1
			outfile.close()

					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()


