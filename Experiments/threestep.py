from ncon import ncon
#import tensorflow.sparse as sp
import numpy as np
import copy

#def sparserepresentation(Y):
#    T = sp.from_dense(Y)
#    return np.concatenate((np.array(T.indices), np.array(T.values).reshape(-1,1)), axis=1)
#def denserepresentation(Y, s):
#    return np.array(sp.to_dense(sp.SparseTensor(indices=Y[:,:-1], values=Y[:,-1], dense_shape=s)))
    
def g(K, lamb):
    if type(K)==np.ndarray:
        return np.linalg.inv(K+lamb*np.diag(np.ones(len(K))))
    elif type(K)==tuple:
        eigval = K[0]
        regular = lamb*np.ones(len(eigval))
        eigvec = K[1]
        return np.matmul(np.matmul(eigvec, np.diag(1/(eigval+regular))), eigvec.transpose())
    else:
        "type of K not knwon"
	    
def h(K, lamb):
    if type(K)==np.ndarray:
        return np.matmul(K, g(K,lamb))
    elif type(K)==tuple:
        eigval = K[0]
        regular = lamb*np.ones(len(eigval))
        eigvec = K[1]
        return np.matmul(np.matmul(eigvec, np.diag(eigval/(eigval+regular))), eigvec.transpose())
    else:
        "type of K not knwon"
	
	
def leaveouthat(hat):
	res = copy.copy(hat)
	for i in range(len(hat)):
		res[i,i]=0
	for i in range(len(hat)):
		res[i,:]=res[i,:]/(1-hat[i,i])
		return res
	
def apply_leave_out_group(leaveoutgroup,H,Pt):
	if len(leaveoutgroup) == 1:
		o = leaveouthat(H[leaveoutgroup[0]-1]) #123 indices to 012 indices
		t = [-1,-2,-3]
		t[leaveoutgroup[0]-1] = leaveoutgroup[0]
		#print("leaveoutcontractions",[t, [-leaveoutgroup[0], leaveoutgroup[0]]])
		return ncon([Pt]+[o], [t, [-leaveoutgroup[0], leaveoutgroup[0]]])
	else :
		# definitions
		matrices = [H[i-1] for i in leaveoutgroup] #123 indices to 012 indices
		t = [-1, -2, -3]
		for i in leaveoutgroup:
			t[i-1] = -t[i-1]
		network = [t] + [[-i,i] for i in leaveoutgroup]
		
		#print("leaveoutcontractions", network)
		
		# full resulet without leave out correction
		res = ncon([Pt] + matrices, network)
		
		# calculate correction factor
		factor = np.ones(Pt.shape)
		
		if 1 in leaveoutgroup:
			for i in range(len(Pt[:,1,1])):
				factor[i,:,:] = H[0][i,i]*factor[i,:,:]
		if 2 in leaveoutgroup:
			for i in range(len(Pt[1,:,1])):
				factor[:,i,:] = H[1][i,i]*factor[:,i,:]
		if 3 in leaveoutgroup:
			for i in range(len(Pt[1,1,:])):
				factor[:,:,i] = H[2][i,i]*factor[:,:,i]
		
		correctionterm = (factor*Pt).astype(dtype=np.float64)
		correctiondivisor = (np.ones(Pt.shape, dtype=np.float64)-factor).astype(dtype=np.float64)
		res = (res-correctionterm)/correctiondivisor			
		return res	
		
def apply_leave_in_group(leaveingroup,H,Pt):
	matrices = [H[i-1] for i in leaveingroup]
	t = [-1, -2, -3]
	for i in leaveingroup:
		t[i-1] = -t[i-1]
	network = [t] + [[-i,i] for i in leaveingroup]
		
	#print("leaveincontractions", network)
		
	# full resulet without leave out correction
	res = ncon([Pt] + matrices, network)
	
	return res
	

class ThreeStep:
	A_=0
	def fit(self, K, Y, lamb):
		if Y.shape == tuple([len(K[i]) for i in range(len(K))]):
				matrices = [Y] + [g(K[i],lamb[i]) for i in range(len(K))]
				self.A_ = ncon(matrices, [[1,2,3], [-1,1],[-2,2],[-3,3]])
	def predict(self, K):
		matrices = [self.A_] + K
		return ncon(matrices, [[1,2,3], [-1,1],[-2,2],[-3,3]])
	
	def leave_out_estimate(self, K,Y, lamb, setting):
		H = [h(K[i], lamb[i]) for i in range(len(K))]
		P = copy.deepcopy(Y)
		leaveingroup = [1,2,3]
		for leaveoutgroup in setting:
			#print(leaveoutgroup)
			for x in leaveoutgroup:
				leaveingroup.remove(x)
			P = apply_leave_out_group(leaveoutgroup, H, P)
			
		P = apply_leave_in_group(leaveingroup, H, P) 
			
			
		return P
		
	
