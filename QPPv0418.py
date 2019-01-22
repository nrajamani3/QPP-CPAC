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
    """This code is adapted from the paper
       "Quasi-periodic patterns(QP):Large-scale dynamics in resting state fMRI that correlate"\
       with local infraslow electrical activity" Shella Keilholz,D et al.NeuroImage,Volume 84, 1 January 2014."\
       The paper implemnts the algorithms for finding QPP in resting state fMRI using matlab"\
       This project is an attempt to adopt the algorithm in python, and to integrate into C-PAC.
       Input:
       ------
       B: 2D nifti image 
       msk: mask of the 2D nifti image
       nd: number of subjects*number of runs per subject
       wl: window length
       nrp: number of repetitions 
       cth: threshold
       n_itr_th: number of iterations
       mx_itr: maximum number of repetitions 
       pfs: path to save the template, FTP, ITP and iter files
       
       
       Returns:
       -------
       time_course_file: 2D array of time points where QPP is detected in .npy format
       ftp_file: 1D array of Final Time Points in .npy format
       itp_file: 1D array of Final Time points in .npy format
       iter_file: 1D array of iterations in .npy format 
       
       Notes:
       -----
       i) If using a .mat file as an input, save only the image with flag 'v7.0' to make it scipy.io loadmat compatible
       (This functionality will soon be replaced by importing with NifTi format only)
       
       ii) To show the peaks found in the signal, add a show=True boolean values in the "find peaks" command.
       A "True" value plots the peaks that are found in the signal.
       
       Examples:
       --------
       >> python detectqppv.py '/path/to/Data/file.mat'
       'path/to/mask/file/' 30 6 0.2 0.3 1 15 'path/to/save/results/' 6 1
    """
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

    #itp = np.random.RandomState(seed=42).permutation(itp)
    itp = itp[0:nrp]

    #Initialize the time course that will later on be saved
    time_course=np.zeros((nrp,nT))

    for irp in range(nrp):
        #initialize a matrix c which will hold the templates
        c=np.zeros(nT)
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
        peaks_size = peaks.size

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
            peaks=detect_peaks(c,mph=cth[1],mpd=wl)
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

        ftp = np.zeros((nrp, peaks.shape[0]))
        iter = np.zeros(nrp)
        if peaks_size>1:
            time_course[irp,:]=c
            ftp[irp]=peaks
            iter[irp]=itr
    #save everything!!


    mdict = {}
    mdict["C"] = time_course
    mdict["FTP"] = ftp
    mdict["ITER"] = iter
    mdict["ITP"] = itp
    #time_course_reshaped= np.reshape(time_course,((4,4,73,61)))
    np.savez('out_array_reshape', time_course)
    nifti_c = nib.Nifti1Image(time_course,np.eye(4))
    nib.save(nifti_c, 'nifti_c.nii')
    #print(time_course_reshaped[20:30])

    np.save('time_course_file',time_course)
    np.save('ftp_file',ftp)
    np.save('iter_file',iter)
    np.save('itp_file',itp)




    return time_course,ftp,itp,iter


def z(x,y):
    x =x[:]-np.mean(x[:])
    y =y[:]-np.mean(y[:])
    if np.norm(x)==0 or np.norm(y)==0:
        z = nan
    else:
        x_trans= np.transpose(X)
        z = (x_trans*y)/norm(x)/norm(y)
    return z

def BSTT(time_course,ftp,nd,nt,nT):
    #load the important files into this

    nRp = time_course.shape[0]

    scmx = np.zeros((nRp,1))
    for i in range(nRp):
        if np.any(ftp[i] != 0):
            p = ftp[i,:]
            d = p.tolist()
            #check out the list index out of range error that pops up
            scmx[i] =np.sum(time_course[i,int(d[i])])
    isscmx = np.argsort(scmx)[::-1]
    T1 = isscmx[0]
    C_1 = time_course[T1,:]

    FTP1 = ftp[T1] #this is a 2D array

    FTP1_flat = FTP1.flatten()
    ftp_list = FTP1_flat.tolist()

    Met1 = np.empty(3)
    for y in range(len(ftp_list)):
        Met1[0] = np.median(C_1[:,int(ftp_list[y])]) #trying to do C_1[:,2d array]
    Met1[1] = np.median(np.diff(FTP1))
    Met1[2] = len(FTP1)
    # plots
    # QPP correlation timecourse and metrics
    C_1_plt = C_1.flatten()
    plt.plot(C_1_plt,'b')
    #plt.plot(C_1_plt[FTP1_flat],'r')
    plt.axis([0,nd*nt,-1,1])
    plt.xticks(np.arange(nt,nT,step=nt))
    plt.yticks(np.arange(-1,1,step=0.2))
    plt.show()

    return C_1,FTP1,Met1

def TBLD2WL(B,wl,FTP1):
    nT = B.shape[1]  # smaller value
    nX = B.shape[0]
    nFTP = len(FTP1)
    WLhs0=round(wl/2)

    WLhe0= WLhs0-np.remainder(wl,2)

    T = np.zeros((nX,2*wl)) #shape is 360,60

    for i in range(nFTP):
        ts=FTP1[:,i]-WLhs0
        te=FTP1[:,i]+wl-1+WLhe0
        te_int = te.astype(int)
        zs = []
        ze = []
        zs = np.array(zs)
        ze = np.array(ze)

        if np.any(ts <= 0):
            ts_int = ts.astype(int)
            zs=np.zeros((nX,abs(ts_int[0])+2))
            ts=1
        if np.any(te>nT):
            ze = np.zeros((nX,abs(te_int[0])-nT+1))
            te=nT


        #    if ts == 1 or te > ts:
        #        temp = np.zeros((nX,te_int[0]-ts))
        #    else:
        #        temp = np.zeros((nX,ts-te_int[0]))
        #    concat_arrays2 = np.concatenate([temp,B[:,ts:te_int[0]]])
        #    T=T+concat_arrays2
        #else:
        if ze.size == 0:
            conct_arrays = np.concatenate((zs,B[:,ts:te_int[0]]),axis=1)
            T = T + conct_arrays

        else:
            conct_arrays = np.concatenate([zs,ze])
            conct_arrays2 = np.concatenate([conct_arrays2,B[:,ts:te_int[0]+1]])
            T = T+conct_arrays2
    T=T/nFTP

    return T

def regressqpp(B,nd,T1,C_1):
    #to do: check shape of c in loop
    wl = np.round(T1.shape[1]/2)
    wlhs = np.round(wl/2)
    wlhe=np.round(wl/2)+wl
    T1c=T1[0,wlhs:wlhe]
    T1c_new =T1[:,wlhs:wlhe]
    nX = B.shape[0]
    nT = B.shape[1]
    nt = nT/nd
    nTf = (nX * wl)
    Br=np.zeros((nX,nT))
    for i in range(nd):
        ts=(i)*nt
        c = C_1[0,ts:ts+nt]
        #c=c.reshape(1,-1)
        # c's shape is now (1,1200)-2D, have to make it a 1D array
        #c=c.ravel()

        for ix in range(nd):
            x = np.convolve(c,T1c,mode='valid')
            y = B[ix,ts+wl-1:ts+nt]
            x_dot = np.dot(x,x)
            y_dot = np.dot(x,y)
            #beta=np.linalg.solve(x_dot,y_dot)
            beta=y_dot/x_dot
            Br[ix,ts+wl-1:ts+nt]=y-x*beta

    C1r=np.zeros((1,nT))
    ntf=nX*wl
    T=np.array(T1c_new.reshape(T1c_new.shape[0]*T1c_new.shape[1]))
    T=T-np.sum(T)/nTf
    t_dot = np.dot(T,T)
    T=T/np.sqrt(t_dot)
    T1n = T.reshape(1,-1)

    for i in range(nd):
        ts=(i)*nt
        for ich in range(nt-wl):
            T = Br[:,ts+ich:ts+ich+wl]
            T=T.flatten()
            T=T-np.sum(T)/ntf
            bch = T/np.sqrt(np.dot(T,T))
            C1r[:,ts+ich]=np.dot(T1n,bch)
    C_1_plt = C_1.flatten()
    C1r_plt = C1r.flatten()
    np.savetxt('regressor file',C1r)
    #np.save('regressor file',C1r)
    plt.plot(C_1_plt,'b')
    plt.plot(C1r_plt,'r')
    plt.axis([0,nd*nt,-1,1])
    plt.xticks(np.arange(nt,nT,step=nt))
    plt.yticks(np.arange(-1,1,step=0.2))
    plt.show()

    return Br, C1r
if __name__ == '__main__':

    qppv(d,msk,nd,wl,nrp,cth,n_itr_th,mx_itr,pfs)





