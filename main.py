from utils import *
from ICA import *

if __name__ == '__main__':
	x = np.linspace(0, 1, 1000)
	S = getData(x)
	X = rnm(S.shape[0])@S
	#plot_signals(S)
	#plot_signals(X)
	Xw = whiten(X)
	reconst=[]
	for k, data in enumerate([X, Xw]):
	    W_est = ICA(data, activation, learning_rate=1e-3, iters=500)
	    reconst.append(W_est@data)
	#print(reconst)
	#plot_signals(S)
	#plot_signals(X)
	plot_signals(reconst[-1])