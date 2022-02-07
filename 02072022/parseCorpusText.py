import os
import sys
import getopt
import re



def parseCorpusOnNewLines(inputFile,outputPrefix,stopCount):
	filehd = open(inputFile, "r")
	lines = filehd.readlines() #note, read whole file into string lines
	filehd.close()
	cnt = 0;
	match_count = 0;

	haveOpenedFile = False;
	for line in lines:
		if not re.match(r'^\n',line):
			if not haveOpenedFile:
				outFile = outputPrefix + "_" + "%04d" % cnt + ".txt"
				outhd = open(outFile, "w")
				haveOpenedFile = True;
				cnt = cnt + 1
			outhd.write(line)
		else:
			match_count += 1
			if match_count > stopCount:
				outhd.close()
				haveOpenedFile = False;
				match_count = 0
			
			

def printUsage():
	print()
	print("Usage: " + sys.argv[0] + " -i <inputTextFile> -o <outputFilePrefix> -s <newline stop count>")
	print()
	sys.exit()


def main():
	inputFile = ""
	outputPrefix = ""
	splitOnNewLines = False
	stopCount = 0;

	if(len(sys.argv) == 1):
		printUsage()
	else:
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'i:o:s:')
			for o, a in opts:
				if o == '-i':
					inputFile = a
				if o == '-o':
					outputPrefix = a
				if o == '-s':
					stopCount = int(a)
		
			parseCorpusOnNewLines(inputFile,outputPrefix,stopCount)
			
		except getopt.GetoptError as err:
			print(err) 
				
if __name__ == '__main__':
	main()