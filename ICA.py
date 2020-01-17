from utils import *

def ICA(X, activation_function,learning_rate=1e-3, w_init='nonsingular', batch_size=500, iters=10e3): 
    W = rnm(d=X.shape[0])
    if w_init=='gaussian':
        W = np.random.rand(X.shape[0], X.shape[0])
    
    while iters:
        a = W@X    
        z = activation_function(a)
        dx = W.T@a
        dw = W + (z@dx.T) / X.shape[1]
        W += learning_rate*dw
        iters-=1
        
    return W