import markovify
import os
import sys
import getopt



def printMarkov(inFile,levels,numSentences,numTries,numRuns):
	with open(inFile) as f:
		text = f.read()

	text_model = markovify.Text(text, state_size=levels)

	for i in range(numRuns):
		outfile = "markov_" + "%04d" % i + ".txt"
		outfh = open(outfile, "w")
		for k in range(numSentences):
			outfh.write(text_model.make_sentence(tries=numTries) + "\n")
		outfh.close()

def printUsage():
	print("python3 markov_simple.py -f <corpus> -n <levels> -N <num sentences> -t <num tries> -r <num runs>")
	sys.exit()

def main():
	
	
	inFile = ""
	levels = 1
	numSentences = 3
	numTries = 100
	numRuns = 1

	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'f:n:N:t:r:')
			for o, a in opts:
				if o == '-f':
					inFile = a
				if o == '-n':
					levels = int(a)
				if o == '-N':
					numSentences = int(a)
				if o == '-t':
					numTries = int(a)
				if o == '-r':
					numRuns = int(a)

			

			
			printMarkov(inFile,levels,numSentences,numTries,numRuns)
		
					
		except getopt.GetoptError as err:
			print(err)


if __name__ == '__main__':
	main()