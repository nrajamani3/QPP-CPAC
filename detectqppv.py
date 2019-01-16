
import h5py
import numpy as np
import os
import nibabel as nib
from scipy import stats
from scipy import io
from QPPv0418 import qppv


def merger(d2,d3,d4,d5,nsubj,nrn):

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


def detect(img_affine,D,D_list,msk,wl,nrp,cth,n_itr_th,mx_itr,pfs,nsubj,nrn):


    if  msk.endswith('.nii'):
        #data1 = nib.load(args.d)
        #data1 = np.array(data1.dataobj) #(61,73,61,8)
        #import msk
        msk_file =nib.load(args.msk)
        msk_img = np.array(msk_file.dataobj)

        #same for masks
        msk_shape = msk_img.shape[:-1]
        m_voxels = np.prod(msk_img.shape[:-1])
        msk = msk_img.reshape(m_voxels,msk_img.shape[-1])

    if  msk.endswith('.mat'):  #we have to remove this, only keeping this for testing
        msk_file = h5py.File(msk)
        msk_img = msk_file['M']
        msk_img = np.array(msk_img)
        msk_shape = msk_img.shape[:-1]
        m_voxels = np.prod(msk_img.shape[:-1])
        msk = msk_img.reshape(m_voxels,msk_img.shape[-1])


    nx = D.shape[3]
    nt = D.shape[2]
    nsubj = D.shape[0]
    nrn = D.shape[1]

    nd = nsubj*nrn
    nt_new = nt * nd
    B = np.zeros((nx,nt_new))
    id =1
    for isbj in range(nsubj):
        for irn in range(nrn):
                D[isbj,irn] = D_list[id-1]
                B[:,(id-1)*nt:id*nt] = np.transpose(stats.zscore(D[isbj,irn]))
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

    #generate qpp
    time_course, ftp, itp, iter = qppv(img_affine,B, msk, nd, wl, nrp, cth, n_itr_th, mx_itr, pfs)
    #choose best template
    FTP1,C_1,Met1 = BSTT(time_course,ftp)
    #regress QPP
    T =TBLD2WL(B,wl,FTP1)
    Br, C1r=regressqpp(B, nd, T, C_1)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()


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

    img_005004 = nib.load('/home/nrajamani/Downloads/0050004_bet.nii')
    img_005004_array = np.array(img_005004.dataobj)
    img_affine = img_005004.affine
    print(img_005004_array[50:60])

    d2 = os.path.join('/home/nrajamani/Downloads/', '0050004.mat')
    d3 = os.path.join('/home/nrajamani/Downloads/', '0050005.mat')
    d4 = os.path.join('/home/nrajamani/Downloads/', '0050006.mat')
    d5 = os.path.join('/home/nrajamani/Downloads/', '0050007.mat')
    D,D_list = merger(d2, d3, d4, d5,args.nsubj,args.nrn)


    detect(img_affine,D,D_list,args.msk,args.wl,args.nrp,args.cth,args.n_itr_th,args.mx_itr,args.pfs,args.nsubj,args.nrn)



