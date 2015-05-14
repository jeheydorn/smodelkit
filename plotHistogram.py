import matplotlib
import matplotlib.pyplot as pyplot
import csv
import sys

def createPlot(xList, xLabel, yLabel, title):
	pyplot.hist(xList, bins=50)
 	pyplot.title(title)
	pyplot.xlabel(xLabel)
	pyplot.ylabel(yLabel)

if __name__ == "__main__":
	inFileName = sys.argv[1]
	inFileNameWithoutExtension = sys.argv[1].split(".")[0]
	legendNames = []
	with open(inFileName, 'rb') as csvFile:
		lines = csvFile.readlines()
		xLabel = lines[0].split(",")[0]
		yLabel = lines[0].split(",")[1]
		lines = lines[1:]
		for i in range(len(lines[0].split(","))):
			legendNames.append(str(i))
			xList = [float(line.split(",")[i]) for line in lines]
			createPlot(xList, xLabel, yLabel, inFileNameWithoutExtension)
	pyplot.legend(legendNames, loc='upper right')
	outFileName = "{0}.png".format(inFileNameWithoutExtension)
	pyplot.savefig(outFileName)
	pyplot.show()







