import numpy
import matplotlib.pyplot as plt
import csv

def load(filename, row = 50):
	"""Load the file named filename

	Keyword arguments:
	filename -- the name of the file to be loaded
	row -- the number of rows 
	"""
	# we create a matrix V in which we will stock the improvement of each step
	V = numpy.zeros((50, 2))
	data_file = open(filename)
	data_frame = csv.reader(data_file, delimiter = ' ')

	for idx, r in enumerate(data_frame):
		if idx < row:
			V[idx, 0] = int(r[0])
			V[idx, 1] = float(r[1])

	data_file.close()
	return V

##########################################################

if __name__ == '__main__':
	V1 = load('ua.txt')
	V2 = load('ua_bias.txt')

	x1 = V1[:,0]
	y1 = V1[:,1]
	x2 = V2[:,0]
	y2 = V2[:,1]

	p1 = plt.plot(x1, y1, marker = 'o')
	p2 = plt.plot(x2, y2, marker = 'v')

	plt.title("Rate of convergence")
	plt.legend([p1, p2], ["base", "with bias"])
	plt.xlabel("Step")
	plt.ylabel("Improvement")

	plt.show()