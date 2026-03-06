import numpy as np
import matplotlib.pyplot as plt

def get_triads(dk=1,kmax=2,kmin=0):
    if int(kmax/dk)!=kmax/dk:
        print('kmax must divide evenly by dk')
        return

    n=int(kmax/dk)*2+1
    A=np.zeros([n]*4)
    idx=dk*(np.array(np.where(A==0))-(kmax/dk)).astype(int)

    tmp1=((-idx[0]-idx[2])**2+(-idx[1]-idx[3])**2<=kmax**2) & (idx[0]**2+idx[1]**2<=kmax**2) & (idx[2]**2+idx[3]**2<=kmax**2) & ~((idx[0]==idx[2])&(idx[1]==idx[3]))
    tmp2=((-idx[0]-idx[2])**2+(-idx[1]-idx[3])**2>=kmin**2) & (idx[0]**2+idx[1]**2>=kmin**2) & (idx[2]**2+idx[3]**2>=kmin**2)
    triads=idx[:,tmp1&tmp2]

    k=triads[[0,1]]
    p=triads[[2,3]]
    q=-triads[[0,1]]-triads[[2,3]]

    order=lambda x: ((x[0]+kmax/dk)*n+(x[1]+kmax/dk))

    idx=(order(k)<=order(p))&(order(p)<=order(q))
    k=k[:,idx];p=p[:,idx];q=q[:,idx]

    return k,p,q

dk=0.5;kmax=30;kmin=6;
k,p,q=get_triads(dk=dk,kmax=kmax,kmin=kmin)
print('done!')

n=int(kmax/dk)*2+1
conn=np.zeros([n]*2)

counts=np.unique(np.column_stack((k,p,q)),axis=1,return_counts=True)
tmp=((counts[0]+kmax)/dk).round().astype(int)
conn[tmp[0],tmp[1]]=counts[1]

fig,ax=plt.subplots(layout='constrained')
im=ax.imshow(conn,extent=[-kmax,kmax,-kmax,kmax],origin='lower')
plt.colorbar(im)
plt.show()
