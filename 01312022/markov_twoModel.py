
import markovify
import os
import sys
import getopt
import spacy
import re


nlp = spacy.blank("en")

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence
        


def getMarkovModel(inFile,levels):
	with open(inFile) as f:
		text = f.read()

	text_model = POSifiedText(text, state_size=levels)
	return text_model


def printMarkov(inFile,inFile2,weightA,weightB,levels,numSentences,numTries):
	

	textModelA = getMarkovModel(inFile,levels)
	textModelB = getMarkovModel(inFile2,levels)

	combinedModel = markovify.combine([ textModelA, textModelB ], [ weightA, weightB ])

	for i in range(numSentences):
		print(combinedModel.make_sentence(tries=numTries))

def printUsage():
	print("python3 markov_twoModel.py -f <corpusOne> -F <corpusTwo> -n <levels> -N <num sentences> -t <num tries> -w <weightOne> -W <weightTwo>")
	sys.exit()

def main():
	
	
	inFile = ""
	inFile2 = ""
	levels = 1
	numSentences = 3
	numTries = 100
	weightA = 1.0
	weightB = 1.0

	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:n:N:t:F:w:W:')
			for o, a in opts:
				if o == '-f':
					inFile = a
				if o == '-F':
					inFile2 = a
				if o == '-n':
					levels = int(a)
				if o == '-N':
					numSentences = int(a)
				if o == '-t':
					numTries = int(a)
				if o == '-w':
					weightA = float(a)
				if o == '-W':
					weightB = float(a)
			

			
			printMarkov(inFile,inFile2,weightA,weightB,levels,numSentences,numTries)
		
					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()