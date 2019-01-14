import scipy.io  
import numpy as np
import nibabel as nib
import os
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
import h5py
from numpy import ndarray
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
#Loading the data file, which is now a matfile. this returns a matlab dictionary with variables names as keys and loaded matrices as values.




def qppv(B,msk,nd,wl,nrp,cth,n_itr_th,mx_itr,pfs):

    #get parameters of the image shape to decide the
    #shape of the cells,arrays,etc
    nT = B.shape[1] #smaller value
    nX = B.shape[0] #larger value
    #print(nT,nX)
    nt = int(nT/nd) #to prevent floating point errors during initializations
    nch = nt-wl+1
    nTf = (nX*wl)
    #make it a boolean mask - all valies with entries greater than zeros will become 1 the rest will be zero
    #no real use of mask anywhere else?
    msk = np.zeros((nX,1))
    msk[(np.sum(abs(B)) > 0)] = 1
    #defining 3D arrayshere. Each array within the 2D array will finally be a nX*wl shape column vector, which will store the template values
    bchf = np.zeros((nT,nX*wl))
    bchfn = np.zeros((nT,nX*wl))
    #for each subject*run store the template into the bchf array. Instead of using transpose and multiplication, just us dot product of the template square to be stored in bchfn,
    #This step. Presumably is done to maximize the peaks that are found within the arrays(eplained below)
    for i in range(nd):
        for ich in range(nch):
            template=B[:,(i)*nt+ich:(i)*nt+wl+ich]
            #change template from a row vector to a column vector
            template = ndarray.flatten(template)
            # insert the template into the bchfn array (this template will be a 1D array)
            bchf[i*nt+ich] = template
            #normalize
            template=template-np.sum(template)/nTf
            #get dot product
            #template_trans = np.transpose(template)
            temp_dot = np.dot(template,template)
            template_sqrt = np.sqrt(temp_dot)
            template=template/template_sqrt
            #add said template into bchfn
            bchfn[(i)*nt+ich] = template
            #removing nan values and making them 0 to prevent further issues in calculations
            A = np.isnan(bchfn)
            bchfn[A] = 0
        #todo: have to make args.nd and other args.something as just the variable name
    #array initialized to later be deleted from the random ITP array
    i2x=np.zeros((nd,wl-1))
    #filling the sequence with range of numbers from wl+2 to nt
    for i in range(1,nd+1):
        i2x[i-1,:] = range(i*nt-wl+2,i*nt+1)

    #delete instances of ITP from i2x
    itp=np.arange(1,nT+1)
    i2x = ndarray.flatten(i2x)
    itp = np.delete(itp,i2x-1,0)
    #permute the numbers within ITP
    itp = np.random.permutation(itp)
    itp = itp[0:nrp]
    #Initialize the time course that will later on be saved
    time_course=np.zeros((nrp,nT))
    ftp = np.zeros((nrp,1))
    iter= np.zeros((nrp,1))
    for irp in range(nrp):
        #initialize a matrix c which will hold the templates
        c=np.zeros(nT)
        #print(c.shape)
        for i in range(nd):
            for ich in range(nch):
                #bchfn_transpose = np.transpose(bchfn[itp[irp]])
                bchfn_1 =bchfn[itp[irp]]
                bchfn_2 =bchfn[i*nt+ich]
                c[(i)*nt+ich]= np.dot(bchfn_1,bchfn_2)
                #print(c.shape)
        #using MARCUS DEUTRE'S awesome detect_peaks.py function which is a replica of the matlab find peaks function
        #switching off show true until it is necessary, in order to test code.
        peaks= detect_peaks(c,mph=cth[0],mpd=wl)
                            #show=True)



#indexes = pu.indexes(c, thresh=c[0])
        #You're deleting the first and last instances of the peaks that are now in the 'peaks' array
        for i in range(nd):
            if i*nt in peaks:
                peaks = np.delete(peaks,np.where(peaks==(i)*nt))
            if i*nt+nch in peaks:
                peaks = np.delete(peaks,np.where(peaks==i*nt+nch))
        #house three copies of templates (inefficient) which is then used to decide between the correlation coefficient in the next loop
        c_0 = c
        c_00 = c
        c_000 = c
        itr = 1
        while itr<=mx_itr:
            c = gaussian_filter(c,0.5)

            if itr<=n_itr_th:
                ith=0
            else:
                ith=1
            th=cth[ith]
            tpsgth=peaks
            n_tpsgth=tpsgth.size
            if n_tpsgth<=1:
                break
            template = bchf[tpsgth[0]]
            for i in range(1,n_tpsgth):
                template=template+bchf[tpsgth[i]]
            template=template/n_tpsgth

            #perform a repeate of the operations in order to find peaks in the template
            #template_trans2=np.transpose(template)
            template=template-np.sum(template)/nTf
            template=template/np.sqrt(np.dot(template,template))
            for i in range(nd):
                for ich in range(nch):
                    c[i*nt+ich]=np.dot(template,bchfn[(i)*nt+ich])
            peaks=detect_peaks(x=c,mph=cth[0],mpd=wl)
            for i in range(nd):
                if i * nt in peaks:
                    peaks = np.delete(peaks, np.where(peaks == (i) * nt))
                if i * nt + nch in peaks:
                    peaks = np.delete(peaks, np.where(peaks == i * nt + nch))
            c_0_norm = (c_0 - np.mean(c_0))/(np.std(c_0))
            #use the correlation coefficient. It returns a matrix and therefore, the first entry of that matrix will be the correlation coefficient value
            if (np.corrcoef(c_0,c)[0,1]>0.9999) or (np.corrcoef(c_00,c)[0,1]>0.9999) or (np.corrcoef(c_000,c)[0,1]>0.9999):
                break

        c_000=c_00
        c_00=c_0
        c_0=c
        itr=itr+1
    if n_tpsgth>1:
        time_course[irp,:]=c
        FTP[[irp]]=tpsgth
        ITER[iRp]=itr
    else:
        pass
    #save everything!!
    mdict = {}
    mdict["C"] = timecourse
    mdict["FTP"] = FTP
    mdict["ITER"] = ITER
    mdict["ITP"] = ITP
    for keys in mdict:
        with open('QPP_results.csv','w') as f:
        data =csv.writer(f)
        mdict = dict(reader)


def z(x,y):
    x =x[:]-np.mean(x[:])
    y =y[:]-np.mean(y[:])
    if np.norm(x)==0 or np.norm(y)==0:
        z = nan
    else:
        x_trans= np.transpose(X)
        z = (x_trans*y)/norm(x)/norm(y)
    return z

def BSTT(file_name=None):
    #load the important files into this
    scipy.loadmat(pf2s,'C','FTP')
    nRp = C.shape[0]
    scmx = np.zeros((nRp,1))
    for i in range(nRp):
        if FTP[i] == 0:
            scmx[i] =np.sum(C[i,FTP[i]])
    isscmx = np.sort(scmx)[::-1]
    iT1 = isscmx[0]
    C_1 = C[iT1,:]
    FTP1 = FTP[iT1]
    Met1 = []
    Met1 = np.array(Met1)
    Met1[0] = np.median(C1[:,FTP1])
    Met1[1] = np.median(no.diff(FTP1))
    Met1[2] = len(FTP1)

def regressqpp(B,nd,T1,C1):
    wl = T1.shape[1]/2
    wlhs = np.round(wl/2)+1
    wlhe=np.round(wl/2)+wl
    T1c=T1[:,wlhs:wlhe]
    nX = B.shape[0]
    nT = B.shape[1]
    nt = nT/nd

    Br=np.zeros((nX,nT))
    for i in range(nd):
        ts=(i-1)*nt
        c = C1[ts+1:ts+nt]
    for ix in range(nX):
        x = np.convolve(c,T1c[ix,:],'valid')
        y = np.transpose(B[ix,ts+wl:ts+nt])
        x_dot = np.dot(x,x)
        y_dot = np.dot(x,y)

        beta=np.linalg.solve(x_dot,y_dot)
        Br[ix,ts+wl:ts+nt]=y-x*beta

    C1r=np.zeros(1,nT)
    ntf=nX*wl
    T=np.flatten(T1c)
    T=T-np.sum(T)/nTf
    T=T/sqrt(np.dot(T,T))
    T1n = np.transpose(T)

    for i in range(nd):
        ts=(i-1)*nt
    for ich in range(nt-wl+1):
        T = Br[:,ts+ich:ts+ich+wl-1]
        T = np.flatten(T)
        T=T-np.sum(T)/nTf
        bch = T/np.sqrt(np.dot(T,T))
        C1r[:,ts+ich]=T1n*bch


if __name__ == '__main__':

    qppv(d,msk,nd,wl,nrp,cth,n_itr_th,mx_itr,pfs)





