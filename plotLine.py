import matplotlib
import matplotlib.pyplot as pyplot
import csv
import sys

def createPlot(xList, yList, xLabel, yLabel, title):
	pyplot.plot(xList, yList)
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
			yList = [float(line.split(",")[i]) for line in lines]
			xList = range(len(yList))
			createPlot(xList, yList, xLabel, yLabel, inFileNameWithoutExtension)
	pyplot.legend(legendNames, loc='upper right')
	outFileName = "{0}.png".format(inFileNameWithoutExtension)
	pyplot.savefig(outFileName)
	pyplot.show()







