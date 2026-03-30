import numpy as np

## Get triads and their magnitudes
def get_triads(duplicates=False,kmax=30,kmin=6):
    n=int(kmax)*2+1
    idk=(np.array(np.where(np.zeros([n]*2)==0))-kmax).astype(int)
    order=lambda x: ((x[0]+kmax)*n+(x[1]+kmax)).astype(int)
    
    k=[];p=[];q=[]
    
    for i in range(len(idk.T)):
        tmp1=(np.linalg.norm(idk.T[i].reshape(2,1),axis=0)<=kmax) & (np.linalg.norm(idk,axis=0)<=kmax) & (np.linalg.norm(-idk-idk.T[i].reshape(2,1),axis=0)<=kmax)
        tmp2=(np.linalg.norm(idk.T[i].reshape(2,1),axis=0)>=kmin) & (np.linalg.norm(idk,axis=0)>=kmin) & (np.linalg.norm(-idk-idk.T[i].reshape(2,1),axis=0)>=kmin)
        if duplicates:
            tmp3=(order(idk.T[i])!=order(idk)) & (order(idk)!=order(-idk-idk.T[i].reshape(2,1))) & (order(idk.T[i].reshape(2,1))!=order(-idk-idk.T[i].reshape(2,1)))
        else:
            tmp3=(order(idk.T[i])<order(idk)) & (order(idk)<order(-idk-idk.T[i].reshape(2,1)))
        if (tmp1&tmp2&tmp3).sum()!=0:
            k.append(np.array([idk.T[i]]*(tmp1&tmp2&tmp3).sum()))
            tmp=idk.T[tmp1&tmp2&tmp3]
            p.append(tmp)
            q.append(-tmp-idk.T[i])
    
    k=np.row_stack(k).T;p=np.row_stack(p).T;q=np.row_stack(q).T
    kmag=np.linalg.norm(k,axis=0);pmag=np.linalg.norm(p,axis=0);qmag=np.linalg.norm(q,axis=0)

    return k,p,q,kmag,pmag,qmag

## Get G*cos*rho (undirected and directed). Can also return triads
def get_G(return_triads=False,return_all_G=False,kmax=30,kmin=6,rho=lambda x:x**(-7/3)):
    n=int(kmax)*2+1
    idk=(np.array(np.where(np.zeros([n]*2)==0))-kmax).astype(int)
    order=lambda x: ((x[0]+kmax)*n+(x[1]+kmax)).astype(int)

    k,p,q,kmag,pmag,qmag=get_triads(duplicates=True,kmax=kmax,kmin=kmin)
    rhok=rho(kmag);rhop=rho(pmag);rhoq=rho(qmag)
    
    K=((qmag**2-pmag**2)*kmag**-2*(rhop*rhoq)**2)+((pmag**2-kmag**2)*qmag**-2*(rhop*rhok)**2)+((kmag**2-qmag**2)*pmag**-2*(rhok*rhoq)**2)
    D=np.sqrt(pmag**2+qmag**2)
    C=-np.cross(q,p,axis=0)/(rhok*rhop*rhoq)*K
    
    dot_term=(k*p).sum(axis=0)
    rho_term=rhok*rhop*rhoq
    cos_term=-C/(2*D)
    term=qmag**2-pmag**2

    Gcosrho=np.zeros([len(idk.T)]*2)
    Gcosrho[order(k),order(p)]=term*np.cross(q,p,axis=0)*cos_term*rho_term
    Gcosrho_dir=np.zeros([len(idk.T)]*2)
    Gcosrho_dir[order(k),order(p)]=np.cross(q,p,axis=0)*cos_term*rho_term*dot_term
    
    if return_all_G:
        G=np.zeros([len(idk.T)]*2)
        G[order(k),order(p)]=term*np.cross(q,p,axis=0)
        Grho=np.zeros([len(idk.T)]*2)
        Grho[order(k),order(p)]=term*np.cross(q,p,axis=0)*rho_term
        Gcos=np.zeros([len(idk.T)]*2)
        Gcos[order(k),order(p)]=term*np.cross(q,p,axis=0)*cos_term
        Gs=[G,Grho,Gcos,Gcosrho]
        
        G=np.zeros([len(idk.T)]*2)
        G[order(k),order(p)]=np.cross(q,p,axis=0)*dot_term
        Grho=np.zeros([len(idk.T)]*2)
        Grho[order(k),order(p)]=np.cross(q,p,axis=0)*rho_term*dot_term
        Gcos=np.zeros([len(idk.T)]*2)
        Gcos[order(k),order(p)]=np.cross(q,p,axis=0)*cos_term*dot_term
        Gs_dir=[G,Grho,Gcos,Gcosrho_dir]
        return k,p,q,kmag,pmag,qmag,Gs,Gs_dir

    if return_triads:
        return k,p,q,kmag,pmag,qmag,Gcosrho,Gcosrho_dir

    return Gcosrho,Gcosrho_dir
