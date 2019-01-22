
import h5py
import numpy as np
import os
import nibabel as nib
from scipy import stats
import scipy.io
from QPPv0418 import qppv,BSTT,TBLD2WL,regressqpp
import time


def merger(d2,d3,d4,d5,nsubj,nrn):
    #if you're giving the images in .nii format, and individual ones
    d2_file = h5py.File(d2)
    d2 = d2_file['B_2']
    d2 = np.array(d2)

    D2 = d2.reshape(d2.shape[0]*d2.shape[1],d2.shape[2])

    d3_file = h5py.File(d3)
    d3 = d3_file['B_4']
    d3 = np.array(d3)
    D3 = d3.reshape(d3.shape[0]*d3.shape[1],d3.shape[2])

    d4_file = h5py.File(d4)
    d4 = d4_file['B_5']
    d4 = np.array(d4)
    D4 = d4.reshape(d4.shape[0]*d4.shape[1],d4.shape[2])

    d5_file = h5py.File(d5)
    d5 = d5_file['B_6']
    d5 = np.array(d5)
    D5 = d5.reshape(d5.shape[0]*d5.shape[1],d5.shape[2])


    D = np.empty((nsubj,nrn,D2.shape[0],D2.shape[1]))
    D_list = [D2,D3,D4,D5]



    return D,D_list


def detect(img,msk,wl,nrp,cth,n_itr_th,mx_itr,pfs,nsubj,nrn):


    if  msk.endswith('.nii'):
        data1 = nib.load(msk)
        msk_img = np.array(data1.dataobj)
        #import msk

        #reshape for masks
        msk_shape = msk_img.shape[:-1]
        m_voxels = np.prod(msk_img.shape[:-1])
        msk = msk_img.reshape(m_voxels,msk_img.shape[-1])

    else:  #we have to remove this, only keeping this for testing
        msk_file = h5py.File(msk)
        msk_img = msk_file['M']
        msk_img = np.array(msk_img)
        msk_shape = msk_img.shape[:-1]
        m_voxels = np.prod(msk_img.shape[:-1])
        msk = msk_img.reshape(m_voxels,msk_img.shape[-1])
    D_file = scipy.io.loadmat(img)
    for keys in D_file:
        D = D_file['D']
    D = np.array(D)

    nx = D[1,1].shape[0]
    nt = D[1,1].shape[1]
    nsubj = D.shape[0]
    nrn = D.shape[1]

    nd = nsubj*nrn
    nt_new = nt * nd
    B = np.zeros((nx,nt_new))
    id =1
    for isbj in range(nsubj):
        for irn in range(nrn):
                B[:,(id-1)*nt:id*nt] = (stats.zscore(D[isbj,irn]))
                id += 1
    msk = np.zeros((nx,1))
    msk[(np.sum(abs(B)) > 0)] = 1
    A = np.isnan(B)
    B[A] = 0

    with open('b.txt','w') as f:
        for line in B:
            f.write("%s\n" %line)

    #make this as avaiable for user input if they need it
    #nrp = nd
    #cth = [0.2,0.3]
    #n_itr_th =1
    #mx_itr=15
    #pf2 =
    start_time = time.time()
    #generate qpp
    time_course, ftp, itp, iter = qppv(B, msk, nd, wl, nrp, cth, n_itr_th, mx_itr, pfs)
    #choose best template
    C_1,FTP1,Met1 = BSTT(time_course,ftp)
    #regress QPP

    T =TBLD2WL(B,wl,FTP1)
    Br, C1r=regressqpp(B, nd, T, C_1)
    print("-----%s seconds ----"%(time.time() - start_time))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("img", type=str,help='Provide the path to the 2D nifti file')
    parser.add_argument("msk", type=str, help='provide the path to the mask of 2D nifti file')


    parser.add_argument("wl", type=int,help='provide the length of window you would like to search for the template in')

    parser.add_argument("nrp", type=int, help='provide the number of random permutations you would like to perform')

    parser.add_argument("cth", nargs= '+',type=float,help='provide the threshold value, as a list')

    parser.add_argument("n_itr_th", type=int, help='provide the number of scans contatenated')

    parser.add_argument("mx_itr", type=int, help='provide the maximum number of iterations')

    parser.add_argument("pfs", type=str, help='provide the path to the directory you would like to save the files in')
    parser.add_argument("nsubj", type=int, help='provide the number of subjects')
    parser.add_argument("nrn",type=int,help='provide the number of runs per subject')
    args = parser.parse_args()



    detect(args.img,args.msk,args.wl,args.nrp,args.cth,args.n_itr_th,args.mx_itr,args.pfs,args.nsubj,args.nrn)



