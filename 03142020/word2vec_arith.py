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
	text = re.sub(r'^\s+', '', text)
	text = re.sub(r'\s+$', '', text)
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
				if wrd.lower() not in stop_words and (myWrdPos[0][1] == 'NN' or myWrdPos[0][1] == 'JJ' or re.search("^V", myWrdPos[0][1])):
					aSentence = aSentence + " " + wrd.lower()
			aSentence = re.sub(r'^\s+', '', aSentence)
			outFH.write(preprocess_text(aSentence.lower())+"\n")
	else:
		for sentence in nltk.sent_tokenize(all_text):
			outFH.write(preprocess_text(sentence.lower())+"\n")

def train_word2vec(filename):
	train_filename = "train_" + filename
	data = gensim.models.word2vec.LineSentence(train_filename)
	return Word2Vec(data, vector_size=200, window=5, min_count=1, workers=multiprocessing.cpu_count())

def train_word2vec_text8():
	corpus = api.load("text8")
	return Word2Vec(corpus, vector_size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())

def getWordEmbeddings(myModel,wrdList,neighborCnt):

	embedding_clusters = []
	word_clusters = []
	for word in wrdList:
		embeddings = []
		words = []
		for similarWrd, _ in myModel.wv.most_similar(word,topn=neighborCnt):
			words.append(similarWrd)
			embeddings.append(myModel[similarWrd])
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
		print("SC: ", score,smin,smax);
		nscore = (2*np.pi)* ((score - smin)/(smax-smin))
		norm.append(nscore)
	return norm

def getTopNNeighbors(myModel,myWord,neighborCnt):
	words = []
	scores = []
	norms = []
	for similarWrd, score in myModel.wv.most_similar(myWord,topn=neighborCnt):
		words.append(similarWrd)
		scores.append(score)

	for word, score in zip(words[:5],scores[:5]):
		print(myWord,word,score)


def getAddNeighbor(myModel,myWord,myWord2,neighborCnt):
	vec1, vec2 = myModel.wv.get_vector(myWord), myModel.wv.get_vector(myWord2)
	result = myModel.wv.similar_by_vector(vec1+vec2);
	returnWord = ''
	for resVal in result:
		if resVal[0] != myWord and resVal[0] != myWord2:
			returnWord = resVal[0]
			break

	return returnWord

def getSubtractNeighbor(myModel,myWord,myWord2,neighborCnt):
	vec1, vec2 = myModel.wv.get_vector(myWord), myModel.wv.get_vector(myWord2)
	result = myModel.wv.similar_by_vector(vec1-vec2);
	returnWord = ''
	for resVal in result:
		if resVal[0] != myWord and resVal[0] != myWord2:
			returnWord = resVal[0]
			break

	return returnWord


def getMultiplyNeighbor(myModel,myWord,myWord2,neighborCnt):
	vec1, vec2 = myModel.wv.get_vector(myWord), myModel.wv.get_vector(myWord2)
	result = myModel.wv.similar_by_vector(vec1*vec2);
	returnWord = ''
	for resVal in result:
		if resVal[0] != myWord and resVal[0] != myWord2:
			returnWord = resVal[0]
			break

	return returnWord

def getDivideNeighbor(myModel,myWord,myWord2,neighborCnt):
	vec1, vec2 = myModel.wv.get_vector(myWord), myModel.wv.get_vector(myWord2)
	result = myModel.wv.similar_by_vector(vec1/vec2);
	returnWord = ''
	for resVal in result:
		if resVal[0] != myWord and resVal[0] != myWord2:
			returnWord = resVal[0]
			break

	return returnWord

def getMidPointNeighbor(myModel,myWord,myWord2,neighborCnt):

	vec1, vec2 = myModel.wv.get_vector(myWord), myModel.wv.get_vector(myWord2)

	result = myModel.wv.similar_by_vector((vec1 + vec2)/2.0)
	#return the first value that isn't myWord or myWord2
	returnRes = ""
	for resVal in result:
		if resVal[0] != myWord and resVal[0] != myWord2:
			returnRes = resVal[0]
			break

	return returnRes


def getAllWordEmbeddings(myModel):
	embeddings = []
	words = []

	for wrd in list(myModel.wv.vocab):
		embeddings.append(myModel.wv[wrd])
		words.append(wrd)

	return embeddings,words


def analogy(myModel,myWord,myWord2,myWord3):
	return myModel.wv.most_similar(positive = [myWord,myWord2], negative = [myWord3])[0][0];
	
def plotSimilarityCircle(arithMode,words,scores,myWord,stride,offset,myWord1,myWord2,myWord3,i):
	plt.figure(figsize=(9, 9))
	colors = cm.rainbow(np.linspace(0, 1, len(words)))
	equation = ""
	if(arithMode == "add"):
		equation = "(" + myWord1 + " " + "+" + " " + myWord2 + ")" + " " + "=" + " " + myWord
	if(arithMode == "subtract"):
		equation = "(" + myWord1 + " " + "-" + " " + myWord2 + ")" + " " + "=" + " " + myWord
	if(arithMode == "divide"):
		equation = "(" + myWord1 + " " + "/" + " " + myWord2 + ")" + " " + "=" + " " + myWord
	if(arithMode == "multiply"):
		equation = "(" + myWord1 + " " + "*" + " " + myWord2 + ")" + " " + "=" + " " + myWord
	if(arithMode == "midpoint"):
		equation = "(" + myWord1 + " " + "+" + " " + myWord2 + ")" + " " + "/" + " " + "2.0" + " " + "=" + " " + myWord
	if(arithMode == "negation"):
		equation = "(" + myWord1 + " " + "+" + " " + myWord2 + ")" + " " + "-" + " " + myWord3 + " " + "=" + " " + myWord
	
	txt = plt.text(-1.0*len(equation)*0.013,0,equation,c="white",size=16)
	txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),path_effects.Normal()])
	#plt.annotate(myWord, alpha=1.0, xy=(0,0), xytext=(5,2), textcoords='offset points', ha='right', va='bottom', size=16)
	cnt = 0
	nwords = []
	nscores = []
	norms = []
	ncolors = []

	# collects words and scores across the stride
	#and normalize again

	for word, score, color in zip(words,scores,colors):
		if cnt % stride == 0:
			nwords.append(word)
			nscores.append(float(cnt)/360.)
		cnt += 1

	norms = normalizeScores(nscores)
	ncolors = cm.rainbow(np.linspace(0, 1, len(nwords)))
	for word, norm, color in zip(nwords[:len(nwords)-1],norms[:len(norms)-1],ncolors[:len(ncolors)-1]):
			x = np.cos(norm)
			y = np.sin(norm)
			plt.scatter(x, y, c=color, alpha=0.0, label=word)
			xoffset = offset*np.cos(norm)
			yoffset = offset*np.sin(norm)
			txt = plt.text(x+xoffset,y+yoffset,word,c=color,size=12)
			txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),path_effects.Normal()])

	plt.axis('off')
	filename = "%04d" % i + ".png"
	plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
	print("Wrote file: ",filename)
	#plt.show()


def getWrdList(fname):
	wlist = []
	fd = open(fname, "r", encoding='utf-8')
	wlist = fd.readlines()
	fd.close()
	wlist = [wrd.rstrip() for wrd in wlist]
	return wlist

def printUsage():
	print("|(Circle Similarity Plot One Word)| python3 word2vec_viz.py -f <inFile> -n <neighbor count> -W <word> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	print("|(Circle Similarity Plot Two Words)| python3 word2vec_viz.py -f <inFile> -n <neighbor count> -W <word> -w <word2> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	print("|(Arithmetic Plot)| python3 word2vec_viz.py -f <inFile> -n <neighbor count> -W <word> -w <word2> -N <negate word> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	print("or for using pre-trained \"text8\" corpus:")
	print("|(Circle Similarity Plot One Word)| python3 word2vec_viz.py -P -n <neighbor count> -W <word> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	print("|(Circle Similarity Plot Two Words)| python3 word2vec_viz.py -P -n <neighbor count> -W <word> -w <word2> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	print("|(Arithmetic plot)| python3 word2vec_viz.py -P -n <neighbor count> -W <word> -w <word2> -N <negate word> -S <useStopWords> -s <angle_stride> -o <negative offset>")
	
	sys.exit()

def main():
	
	
	inFile = ""
	wordList = ""
	neighborCnt = 30
	prplx = 30
	useStopWords = True
	arithMode = "none"
	
	usePreTrained = False

	myWord = ""
	myWord2 = ""
	myWord3 = ""
	words = []
	words2 = []
	scores = []
	scores2 = []
	embeddings = []
	embedding_clusters = []
	embeddings_2d = []
	stride = 22
	offset = 0
	itr = 1
	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:n:W:w:s:N:i:TASMDPY')
			for o, a in opts:
				if o == '-f':
					inFile = a
				if o == '-n':
					neighborCnt = int(a)
				if o == '-W':
					myWord = a
				if o == '-w':
					myWord2 = a
				if o == '-N':
					myWord3 = a
				if o == '-s':
					stride = int(a)
				if o == '-i':
					itr = int(a)
				if o == '-P':
					usePreTrained = True
				if o == '-T':
					arithMode = "midpoint"
				if o == '-A':
					arithMode = "add"
				if o == '-S':
					arithMode = "subtract"
				if o == '-M':
					arithMode = "multiply"
				if o == '-D':
					arithMode = "divide"
				if o == '-Y':
					arithMode = "negation"

			#parse input file for training
			#train word2vec model
			if usePreTrained:
				myModel = train_word2vec_text8()
			else:
				make_trainData(inFile,useStopWords)
				myModel = train_word2vec(inFile)

			if(arithMode == "add"):
				for i in range(0,itr):
					resultWrd = getAddNeighbor(myModel,myWord,myWord2,neighborCnt)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)
					myWord2 = myWord
					myWord = resultWrd
			if(arithMode == "subtract"):
				for i in range(0,itr):
					resultWrd = getSubtractNeighbor(myModel,myWord,myWord2,neighborCnt)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)
					myWord2 = myWord
					myWord = resultWrd
			if(arithMode == "multiply"):
				for i in range(0,itr):
					resultWrd = getMultiplyNeighbor(myModel,myWord,myWord2,neighborCnt)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)
					myWord2 = myWord
					myWord = resultWrd
			if(arithMode == "divide"):
				for i in range(0,itr):
					resultWrd = getDivideNeighbor(myModel,myWord,myWord2,neighborCnt)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)
					myWord2 = myWord
					myWord = resultWrd
			if(arithMode == "midpoint"):
				for i in range(0,itr):
					resultWrd = getMidPointNeighbor(myModel,myWord,myWord2,neighborCnt)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)
					myWord2 = myWord
					myWord = resultWrd
				
			if(arithMode == "negation"):
				for i in range(0,itr):
					resultWrd = analogy(myModel,myWord,myWord2,myWord3)
					words, scores = getNeighborSimilarity(myModel,resultWrd,neighborCnt)
					plotSimilarityCircle(arithMode,words,scores,resultWrd,stride,offset,myWord,myWord2,myWord3,i)	
					myWord3 = myWord2
					myWord2 = myWord
					myWord = resultWrd		
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()


