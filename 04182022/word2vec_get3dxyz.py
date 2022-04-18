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
	stop_words = []
	all_text = open(inFile, "r", encoding='utf-8').read()
	outFile = "train_" + inFile
	outFH = open(outFile, "w", encoding='utf-8')

	if(useStopWords):
		stop_words = set(nltk.corpus.stopwords.words('english'))
		for sentence in nltk.sent_tokenize(all_text):
			aSentence = ""
			for wrd in nltk.word_tokenize(sentence):
				myWrdPos = nltk.pos_tag(nltk.word_tokenize(wrd))
				#if wrd.lower() not in stop_words and (myWrdPos[0][1] == 'NN' or myWrdPos[0][1] == 'JJ' or re.search("^V", myWrdPos[0][1])):
				aSentence = aSentence + " " + wrd
			aSentence = re.sub(r'^\s+', '', aSentence)
			outFH.write(preprocess_text(aSentence)+"\n")
	else:
		for sentence in nltk.sent_tokenize(all_text):
			outFH.write(preprocess_text(sentence)+"\n")

def train_word2vec(filename):
	train_filename = "train_" + filename
	data = gensim.models.word2vec.LineSentence(train_filename)
	print("TWV: ",data)
	return Word2Vec(data, vector_size=200, window=5, min_count=1, workers=multiprocessing.cpu_count())

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
					if wrd not in concord:
						concord[preprocess_text(wrd)] = 1
					else:
						concord[preprocess_text(wrd)] += 1
			
	return concord

def get_ProperNouns(inFile,useStopWords):
	properNouns = set()
	stop_words = []
	all_text = open(inFile, "r", encoding='utf-8').read()

	maxFreq = -9999999;
	if(useStopWords):
		stop_words = set(nltk.corpus.stopwords.words('english'))
		for sentence in nltk.sent_tokenize(all_text):
			for wrd in nltk.word_tokenize(sentence):
				myWrdPos = nltk.pos_tag(nltk.word_tokenize(wrd))
				if (myWrdPos[0][1] == 'NNP'):
					properNouns.add(preprocess_text(wrd))

	print("PN:",properNouns)
					
			
	return properNouns

def getMaxCountWords(concord,maxFreq):
	maxCountWords = []
	for wrd in concord:
		if(concord[wrd] >= maxFreq):
			maxCountWords.append(wrd)
	return maxCountWords

def getMinCountWords(concord,minFreq):
	minCountWords = []
	for wrd in concord:
		if(concord[wrd] <= minFreq and concord[wrd] > 1):
			minCountWords.append(wrd)
	return minCountWords

def getRangeCountWords(concord,minFreq,maxFreq):
	rangeCountWords = []
	for wrd in concord:
		if(concord[wrd] >= minFreq and concord[wrd] <= maxFreq):
			rangeCountWords.append(wrd)
	return rangeCountWords


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
			print("(pretrained) WORD not found",word)
	return embeddings, foundWordList

def getWordEmbeddingsTrained(myModel,wrdList):
	embeddings = []
	foundWordList = []
	for word in wrdList:
		if word in myModel.wv:
			embeddings.append(myModel.wv[word])
			foundWordList.append(word)
		else:
			print("(trained) WORD not found",word)
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
	preTrainedWordVectors = "NONE"
	neighborCnt = 30
	prplx = 30
	useStopWords = True
	useCorpus = False
	useWordList = False
	useTrainModel = False
	useProperNouns = False
	embeddings = []
	embeddings_3d = []
	
	myWrdList = []
	foundWordList = []
	myConcord = {}
	maxFreq = 0
	minFreq = 0;
	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:m:P:M:N:TpC')
			for o, a in opts:
				if o == '-f':
					inFile = a
				if o == '-m':
					wordListFile = a
					useWordList = True
				if o == '-M':
					maxFreq = int(a)
				if o == '-N':
					minFreq = int(a)
				if o == '-p':
					useProperNouns = True
				if o == '-P':
					preTrainedWordVectors = a
				if o == '-T':
					useTrainModel = True
				if o == '-C':
					useCorpus = True

				
			
			myConcord = make_Concordance(inFile,useStopWords)
			myModel = getPreTrainedModel(preTrainedWordVectors)
			tsne_3d = TSNE(perplexity=prplx, n_components=3, init='pca', n_iter=3500, random_state=12)
			for i in range(10,150,10):
				minFreq = i
				maxFreq = minFreq + 10	
				myWrdList = getRangeCountWords(myConcord,minFreq,maxFreq)
				print("Words Range: ",minFreq,maxFreq,myWrdList)
				embeddings,foundWordList = getWordEmbeddings(myModel,myWrdList)
				embeddings_3d = tsne_3d.fit_transform(embeddings)
		
			
				cnt = 0;
				ofname = "myWordsXYZ_" + "%04d" % minFreq + ".txt"
				ofname2 = "myMesh_" + "%04d" % minFreq + ".txt"
				outfile = open(ofname, "w")
				outfile2 = open(ofname2, "w")
				header = "cnt,wrd,x,y,z"
				outfile.write(header + "\n")
				for em3d,wrd in zip(embeddings_3d,foundWordList): 
					outfile.write(str(cnt) + "," + wrd + "," + str(em3d[0]) + "," + str(em3d[1]) + "," + str(em3d[2]) + "\n")
					outfile2.write(str(em3d[0]) + "," + str(em3d[1]) + "," + str(em3d[2]) + "\n")
					cnt += 1
				outfile.close()
				outfile2.close()

					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()


