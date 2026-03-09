import numpy as np
import matplotlib.pyplot as plt

dk=0.5;kmax=30;kmin=6
n=int(kmax/dk)*2+1
idk=dk*(np.array(np.where(np.zeros([n]*2)==0))-(kmax/dk)).astype(int)
order=lambda x: ((x[0]+kmax)/dk)*n+((x[1]+kmax)/dk)
kmaxx=kmax**2
kminn=kmin**2

k=[];p=[];q=[]

for i in range(len(idk.T)):
    tmp1=(np.linalg.norm(idk.T[i].reshape(2,1),axis=0)**2<=kmaxx) & (np.linalg.norm(idk,axis=0)**2<=kmaxx) & (np.linalg.norm(-idk-idk.T[i].reshape(2,1),axis=0)**2<=kmaxx)
    tmp2=(np.linalg.norm(idk.T[i].reshape(2,1),axis=0)**2>=kminn) & (np.linalg.norm(idk,axis=0)**2>=kminn) & (np.linalg.norm(-idk-idk.T[i].reshape(2,1),axis=0)**2>=kminn)
    tmp3=(order(idk.T[i])<order(idk)) & (order(idk)<order(-idk-idk.T[i].reshape(2,1)))
    if (tmp1&tmp2&tmp3).sum()!=0:
        k.append(np.array([idk.T[i]]*(tmp1&tmp2&tmp3).sum()))
        tmp=idk.T[tmp1&tmp2&tmp3]
        p.append(tmp)
        q.append(-tmp-idk.T[i])

k=np.row_stack(k).T;p=np.row_stack(p).T;q=np.row_stack(q).T
A=np.zeros([len(idk.T)]*2);G=np.zeros([len(idk.T)]*2)
A[order(k).astype(int),order(p).astype(int)]=1
G[order(k).astype(int),order(p).astype(int)]=(np.linalg.norm(k,axis=0)**2)*np.cross(q,p,axis=0)

print('done!')

conn=np.zeros([n]*2)
counts=np.unique(np.column_stack((k,p,q)),axis=1,return_counts=True)
tmp=((counts[0]+kmax)/dk).round().astype(int)
conn[tmp[0],tmp[1]]=counts[1]

fig,ax=plt.subplots(layout='constrained')
im=ax.imshow(conn,extent=[-kmax,kmax,-kmax,kmax],origin='lower',cmap='plasma')
plt.colorbar(im)
plt.savefig('./img/heatmap.png',dpi=300,bbox_inches='tight')
plt.show()

conn=np.zeros([n]*2)
tmp=((idk+kmax)/dk).round().astype(int)
conn[tmp[0],tmp[1]]=G.sum(axis=1)

fig,ax=plt.subplots(layout='constrained')
im=ax.imshow(conn,extent=[-kmax,kmax,-kmax,kmax],origin='lower')
plt.colorbar(im)
plt.savefig('./img/heatmap2.png',dpi=300,bbox_inches='tight')
plt.show()
