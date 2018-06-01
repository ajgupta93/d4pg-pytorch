import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def main():
    
	file_list = os.listdir('./')
	csv_files = [filename for filename in file_list if filename.split('.')[1] == 'csv']

	for file_name in csv_files:
	    env_name = file_name.split('-')[0]
	    with open(file_name, 'rb') as csvfile:
	        steps = []
	        avg_return = []
	        curr_return = []
	        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	        for row in reader:
	            x = row[0].split(',')
	            steps.append(int(float(x[0])))
	            avg_return.append(float(x[1]))
	            curr_return.append(float(x[2]))
	        fig = plt.figure()
	        out = numpy_ewma_vectorized_v2(np.array(curr_return), 20)
	        plt.plot(steps, out)
	        plt.title("Score vs steps for " + env_name)
	        plt.xlabel("Steps")
	        plt.ylabel("Score")
	        plt.savefig(env_name+".png")
	        plt.close(fig) #plt.show()

if __name__=="__main__":
	main()