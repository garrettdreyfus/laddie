import numpy as np
from physics import *
from tools import *
import pdb
import matplotlib.pyplot as plt
from numba import njit
import tools
import pyamg
from pyamg.krylov import bicgstab

def integrate(object,nsteps=2):
    """Integration of N time steps. During normal integration, nsteps = 2 (now-centered Leapfrog scheme)"""
    intD(object,nsteps*object.dt)
    prepare_integrate(object)
    generate_stars(object,nsteps*object.dt)
    surface_pressure(object,nsteps*object.dt)
    #intU(object,nsteps*object.dt)
    #intV(object,nsteps*object.dt)
    intT(object,nsteps*object.dt)
    intS(object,nsteps*object.dt)        

def prepare_integrate(object):
    """Compute reused fields in integration, after integrating D"""
    object.D2 = (object.H-object.D)*object.tmask
    if (object.D2[1][object.tmask==1]<=0).any():
        print("LAYER THICKNESS WENT TO ZERO")
        exit()
    object.dDdt = (object.D[2,:,:]-object.D[0,:,:]) / (2*object.dt)
    object.dD2dt = (object.D2[2,:,:]-object.D2[0,:,:]) / (2*object.dt)
    object.Ddrho = object.D[1,:,:]*object.drho
    object.TWterm = object.g*(object.zb-object.D[1,:,:])*(object.drho)
       
    if object.convop == 2:
        object.conv2 = np.where(object.drho<0,1,0)*object.D[1,:,:]/object.convtime# *np.where(object.convop==2,1,0)
    else:
        object.conv2 = 0

def timefilter(object):
    """Time filter, Robert Asselin scheme"""
    object.D[1,:,:] += object.nu/2 * (object.D[0,:,:]+object.D[2,:,:]-2*object.D[1,:,:]) * object.tmask
    object.U[1,:,:] += object.nu/2 * (object.U[0,:,:]+object.U[2,:,:]-2*object.U[1,:,:]) * object.umask
    object.V[1,:,:] += object.nu/2 * (object.V[0,:,:]+object.V[2,:,:]-2*object.V[1,:,:]) * object.vmask

    object.D2[1,:,:] = (object.H-object.D[1])*object.tmask
    object.U2[1,:,:] += object.nu/2 * (object.U2[0,:,:]+object.U2[2,:,:]-2*object.U2[1,:,:]) * object.umask
    object.V2[1,:,:] += object.nu/2 * (object.V2[0,:,:]+object.V2[2,:,:]-2*object.V2[1,:,:]) * object.vmask

    object.T[1,:,:] += object.nu/2 * (object.T[0,:,:]+object.T[2,:,:]-2*object.T[1,:,:]) * object.tmask
    object.S[1,:,:] += object.nu/2 * (object.S[0,:,:]+object.S[2,:,:]-2*object.S[1,:,:]) * object.tmask

    update_density(object)
    update_convection(object)

def updatevars(object):
    """Update temporary variables"""
    object.D = np.roll(object.D,-1,axis=0)
    object.D2 = (object.zb-object.B)-object.D
    object.U = np.roll(object.U,-1,axis=0)
    object.V = np.roll(object.V,-1,axis=0)
    object.U2 = np.roll(object.U2,-1,axis=0)
    object.V2 = np.roll(object.V2,-1,axis=0)
    object.T = np.roll(object.T,-1,axis=0)
    object.S = np.roll(object.S,-1,axis=0)

    updatesecondary(object)
    
def cutforstability(object):
    """Cut U, and V when exceeding specified thresholds"""
    object.U[2,:,:] = np.where(object.U[2,:,:]> object.vcut, object.vcut,object.U[2,:,:])
    object.U[2,:,:] = np.where(object.U[2,:,:]<-object.vcut,-object.vcut,object.U[2,:,:])
    object.V[2,:,:] = np.where(object.V[2,:,:]> object.vcut, object.vcut,object.V[2,:,:])
    object.V[2,:,:] = np.where(object.V[2,:,:]<-object.vcut,-object.vcut,object.V[2,:,:])   

    object.U2[2,:,:] = np.where(object.U2[2,:,:]> object.vcut, object.vcut,object.U2[2,:,:])
    object.U2[2,:,:] = np.where(object.U2[2,:,:]<-object.vcut,-object.vcut,object.U2[2,:,:])
    object.V2[2,:,:] = np.where(object.V2[2,:,:]> object.vcut, object.vcut,object.V2[2,:,:])
    object.V2[2,:,:] = np.where(object.V2[2,:,:]<-object.vcut,-object.vcut,object.V2[2,:,:])   

# [X] convD2
# [X] dD2dt
# [X] convU2
# [X] D2xm1
# [X] D2xp1
# [X] D2ym1
# [X] D2xp1
# [X] lapU2
# [X] lapV2
def intD(object,delt):
    """Integrate D. Multipy RHS of dD/dt with delt (= 2x dt for LeapFrog)"""
    object.D[2,:,:] = object.D[0,:,:] \
                    + (object.convD \
                    +  object.melt \
                    +  object.nentr \
                    ) * object.tmask * delt    

def generate_stars(object,delt):
    #hello
    """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.Ustar = object.U[0,:,:] \
                    + div0((-object.U[1,:,:] * ip_t(object,object.dDdt) \
                    +  convU(object) \
                    #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

                    ## PRESSURE TERMS
                    ### --------------------
                    - object.g*ip_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=1)-object.drho)/(object.dx) \
                            
                    ### --------------------

                    +  object.f*ip_t(object,object.D[1,:,:]*object.Vjm) \
                    +  -object.Cd* object.U[1,:,:] *(object.U[1,:,:]**2 + ip(jm(object.V[1,:,:]))**2)**.5 \
                    +  -object.Av* div0((object.U[1,:,:]-object.U2[1,:,:]),ip_t(object,object.H/2.0)) \
                    +  object.Ah*lapU(object) \
                    +  -0*object.detr* object.U[1,:,:] \
                    ),ip_t(object,object.D[1,:,:])) * object.umask * delt

    """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    #plt.imshow(- .5*object.g*jp_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy)

    object.Vstar = object.V[0,:,:] \
                    +div0((-object.V[1,:,:] * jp_t(object,object.dDdt) \
                    + convV(object) \
                    #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #PRESSURE TERMS
                    #-------------------------
                    #- .5*object.g*jp_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    - object.g*jp_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=0)-object.drho)/(object.dy) \
                    #-------------------------

                    + -object.f*jp_t(object,object.D[1,:,:]*object.Uim) \
                    + -object.Cd* object.V[1,:,:] *(object.V[1,:,:]**2 + jp(im(object.U[1,:,:]))**2)**.5 \
                    + -object.Av* div0((object.V[1,:,:]-object.V2[1,:,:]),jp_t(object,object.H/2.0)) \
                    + object.Ah*lapV(object) \
                    +  -0*object.detr* object.V[1,:,:] \
                    ),jp_t(object,object.D[1,:,:])) * object.vmask * delt

    """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.Ustar2 = object.U2[0,:,:] \
                    + div0((-object.U2[1,:,:] * ip_t(object,object.dD2dt) \
                    +  convU2(object) \
                    #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

                    ## PRESSURE TERMS
                    ### --------------------
                    #-  .5*object.g*ip_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D2[1,:,:]))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
                    - ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx)\
                    ### --------------------

                    +  object.f*ip_t(object,object.D2[1,:,:]*object.V2jm) \
                    +  -object.Cd* object.U2[1,:,:] *(object.U2[1,:,:]**2 + ip(jm(object.V2[1,:,:]))**2)**.5 \
                    +  object.Ah*lapU2(object) \
                    + -object.Av* div0((object.U2[1,:,:]-object.U[1,:,:]),ip_t(object,object.H/2.0)) \
                    -  -0*object.detr* object.U2[1,:,:] \
                    ),ip_t(object,object.D2[1,:,:])) * object.umask * delt
    """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.Vstar2 = object.V2[0,:,:] \
                    +div0((-object.V2[1,:,:] * jp_t(object,object.dD2dt) \
                    + convV2(object) \
                    #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #PRESSURE TERMS
                    #-------------------------
                    #- .5*object.g*jp_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    - jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy)\
                    #-------------------------

                    + -object.f*jp_t(object,object.D2[1,:,:]*object.U2im) \
                    + -object.Cd* object.V2[1,:,:] *(object.V2[1,:,:]**2 + jp(im(object.U2[1,:,:]))**2)**.5 \
                    + object.Ah*lapV2(object) \
                    + -object.Av* div0((object.V2[1,:,:]-object.V[1,:,:]),jp_t(object,object.H/2.0)) \
                    -  -0*object.detr* object.V2[1,:,:] \
                    ),jp_t(object,object.D2[1,:,:])) * object.vmask * delt

@njit
def SOR(pi,pi_rhs,Osum,Os,Ow,rp,pi_tol,Nx,Ny,tmask):
    iters =0 
    while True:
        maxdiff =0 
        absdiff=0
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                pi_prev = pi[i,j]
                #pi[i,j] = (1-rp)*pi[i,j] \
                            #+ rp * Osum[i,j] \
                                #*  ( Os[i+1,j]*pi[i+1,j] + Os[i,j]*pi[i-1][j] + Ow[i,j+1]*pi[i,j+1] + Ow[i,j]*pi[i,j-1] - pi_rhs[i,j] );
                pi[i,j] = (1-rp)*pi[i,j] - rp * Osum[i,j]*pi_rhs[i,j]

                if tmask[i+1,j]:
                    pi[i,j] += rp*Osum[i,j]*Os[i,j]*pi[i+1,j]
##
                if tmask[i-1,j]:
                    pi[i,j] += rp*Osum[i,j]*Os[i-1,j]*pi[i-1,j]
#
                if tmask[i,j+1]:
                    pi[i,j] += rp*Osum[i,j]*Ow[i,j]*pi[i,j+1]
##
                if tmask[i,j-1]:
                    pi[i,j] += rp*Osum[i,j]*Ow[i,j-1]*pi[i,j-1]
#
                absdiff = abs(pi_prev-pi[i,j])
                if absdiff>maxdiff:
                    maxdiff=absdiff
        if iters%100000==0:
            print(maxdiff)
        if maxdiff<pi_tol:
            print("SOR: ",iters)
            return pi
        iters+=1

@njit
def assemble_pi_rhs(pi_rhs,hu,hv,dx,dy,dt,umask,vmask,tmask):
    for i in range(1,pi_rhs.shape[0]):
        for j in range(1,pi_rhs.shape[1]):
            if vmask[i,j]!=0 and tmask[i+1,j]!=0:
                pi_rhs[i][j] = pi_rhs[i,j] - hv[i,j]/(dy*dt)
                pi_rhs[i+1][j] = pi_rhs[i+1,j]+ hv[i,j]/(dy*dt)
            if umask[i,j]!=0 and tmask[i,j+1]!=0:
                pi_rhs[i][j] = pi_rhs[i,j] - hu[i,j]/(dx*dt)
                pi_rhs[i][j+1] = pi_rhs[i,j+1] + hu[i,j]/(dx*dt)
    return pi_rhs

@njit
def assemble_mg_b(b,flatindexes,hu,hv,dx,dy,dt,umask,vmask,tmask):
    Ny = hu.shape[0]
    for i in range(0,hu.shape[0]):
        for j in range(0,hu.shape[1]):
            if vmask[i,j]!=0 and tmask[i+1,j]!=0 and tmask[i,j]:
                b[flatindexes[i,j]] = b[flatindexes[i,j]] - hv[i,j]/(dy*dt)
                b[flatindexes[i+1,j]] = b[flatindexes[i+1,j]]+ hv[i,j]/(dy*dt)
            if umask[i,j]!=0 and tmask[i,j+1]!=0 and tmask[i,j]:
                b[flatindexes[i,j]] = b[flatindexes[i,j]] - hu[i,j]/(dx*dt)
                b[flatindexes[i,j+1]] = b[flatindexes[i,j+1]] + hu[i,j]/(dx*dt)
    return b

@njit
def assemble_mg_A(A,H,flatindexes,tmask,dx,dy):
    dx2q=dx**2
    dy2q=dy**2
    Ny = H.shape[0]
    for i in range(1,H.shape[0]-1):
        for j in range(1,H.shape[1]-1):
            if tmask[i,j]:
                if tmask[i,j+1]:
                    A[flatindexes[i,j]][flatindexes[i,j+1]] = -( ((H[i,j+1]+H[i,j])/2.0)/dx2q)
                    A[flatindexes[i,j]][flatindexes[i,j]] -= ( -((H[i,j+1]+H[i,j])/2.0)/dx2q)
                if tmask[i,j-1]:
                    A[flatindexes[i,j]][flatindexes[i,j-1]] = -((H[i,j-1]+H[i,j])/2.0)/dx2q
                    A[flatindexes[i,j]][flatindexes[i,j]] += ((H[i,j-1]+H[i,j])/2.0)/dx2q
                if tmask[i+1,j]:
                    A[flatindexes[i,j]][flatindexes[i+1,j]] =-( ((H[i+1,j]+H[i,j])/2.0)/dy2q)
                    A[flatindexes[i,j]][flatindexes[i,j]] -= ( -((H[i+1,j]+H[i,j])/2.0)/dy2q)
                if tmask[i-1,j]:
                    A[flatindexes[i,j]][flatindexes[i-1,j]] = -((H[i-1,j]+H[i,j])/2.0)/dy2q
                    A[flatindexes[i,j]][flatindexes[i,j]] += ((H[i-1,j]+H[i,j])/2.0)/dy2q
    return A


#SOR_jit = jit(SOR)

#def create_pi_rhs(pi_rhs,hu1,hv1,hu2,hv2,dx,dy,vmask,umask):
    #for i in range:

@njit
def assemble_Osum(H,tmask,dx,dy):
    Osum = np.zeros(H.shape)
    Os = np.zeros(H.shape)
    Ow = np.zeros(H.shape)
    dx2q=dx**2
    dy2q=dy**2
    for i in range(0,H.shape[0]-1):
        for j in range(0,H.shape[1]-1):
            if tmask[i,j+1] and tmask[i,j]:
                h_west=(H[i,j+1]+H[i,j])/2.0
            else:
                h_west=0
            if tmask[i+1,j] and tmask[i,j]:
                h_south=(H[i+1,j]+H[i,j])/2.0
            else:
                h_south =0 
            Ow[i,j]=h_west/dx2q
            Os[i,j]=h_south/dy2q
    for i in range(1,H.shape[0]):
        for j in range(1,H.shape[1]):
            Osum[i,j]+=Ow[i,j]+Os[i,j]
            if tmask[i,j-1]==1:
                Osum[i,j]+=Ow[i,j-1]
            if tmask[i-1,j]==1:
                Osum[i,j]+=Os[i-1,j]

    for i in range(H.shape[0]):
        for j in range(0,H.shape[1]):
            if tmask[i,j]:
                Osum[i,j] = 1.0/Osum[i,j]
            else:
                Osum[i,j] = 0


    return Osum,Os,Ow




def surface_pressure(object,delt,method="mg"):
    #X,Y = np.meshgrid(range(object.nx+2)*object.dx,range(object.ny+2)*object.dy)
    #X = (X-np.max(X)/2)/np.max(X)
    #Y = (Y-np.max(Y)/2)/np.max(Y)
    #Xnew = X/(0.1*(X**2+Y**2))
    #Xnew[np.abs(Xnew)>100]=100*np.sign(Xnew[np.abs(Xnew)>100])
    #Ynew = Y/(0.1*(X**2+Y**2))
    #Ynew[np.abs(Ynew)>100]=100*np.sign(Ynew[np.abs(Ynew)>100])
    #X=Xnew
    #Y=Ynew
    #print("-------")
    ##object.vmask=np.roll(object.vmask,1,axis=0)
    #object.tmask[:]=1
    #object.umask[:]=1
    #object.vmask[:]=1

    #object.H[:]=20
    #object.H[:10,:]=0
    #object.H[-10:,:]=0
    #object.H[:,:10]=0
    #object.H[:,-10:]=0

    #object.tmask[:10,:]=0
    #object.tmask[-10:,:]=0
    #object.tmask[:,:10]=0
    #object.tmask[:,-10:]=0


#
    #object.umask[:10,:]=0
    #object.umask[-10:,:]=0
    #object.umask[:,:11]=0
    #object.umask[:,-10:]=0

    #object.vmask[:11,:]=0
    #object.vmask[-10:,:]=0
    #object.vmask[:,:10]=0
    #object.vmask[:,-10:]=0

    #object.D[:]=10
    #object.D2[:]=10
    #object.D2[1][object.tmask==0]=0
    #object.D[1][object.tmask==0]=0

    #object.Vstar[:] = 0
    #object.Ustar[:] = (-Y/10)#object.D[1]
    #object.Vstar2[:] = 0
    #object.Ustar2[:] = (-Y/10)#object.D2[1]
#
    #object.Vstar[:] = -1#(-Y/10)*object.vmask#object.D[1]
    #object.Ustar[:] = 0
    #object.Vstar2[:] = -1#(-Y/10)*object.vmask#object.D2[1]
    #object.Ustar2[:] = 0
    ##hu1 = object.Ustar*Dw#im_t(object,object.D[1])
    #hv1 = object.Vstar*Ds#jm_t(object,object.D[1])
    #hu2 = object.Ustar2*D2s#im_t(object,object.D2[1])
    #hv2 = object.Vstar2*D2w#jm_t(object,object.D2[1])



    #You have to fix this !!!!
    hu1 = object.Ustar*tools.ip_t(object,object.D[1])
    hv1 = object.Vstar*tools.jp_t(object,object.D[1])
    hu2 = object.Ustar2*tools.ip_t(object,object.D2[1])
    hv2 = object.Vstar2*tools.jp_t(object,object.D2[1])
    debug = False
#

    if (object.pressure_solves)%5==0 and True and debug:

        #termu = - object.g*ip_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=1)-object.drho)/(object.dx)
        #termu[object.umask==0]=np.nan
        #termv = - object.g*jp_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=0)-object.drho)/(object.dy) 
        #termv[object.vmask==0]=np.nan

        #termu2 = - ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx)
        #termu2[object.umask==0]=np.nan
        #termv2 = - jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy)
        #termv2[object.vmask==0]=np.nan
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        im = ax1.imshow(object.Ustar)
        plt.colorbar(im,ax=ax1)
        im = ax2.imshow(object.Vstar)
        plt.colorbar(im,ax=ax2)
        im = ax3.imshow(object.Ustar2)
        plt.colorbar(im,ax=ax3)
        im = ax4.imshow(object.Vstar2)
        plt.colorbar(im,ax=ax4)
        plt.show()





    if object.pressure_solves==0:
        if method=="mg":
            nonzero = np.where(object.tmask!=0)
            object.flatindexes=np.ones(hu1.shape,dtype=int)*-999999999999
            count=0
            for i in range(len(nonzero[0])):
                object.flatindexes[nonzero[0][i],nonzero[1][i]]=count
                count+=1
            A = np.zeros([count,count])
            A = assemble_mg_A(A,object.H,object.flatindexes,object.tmask,object.dx,object.dy)
#
            B = np.ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()
#
            ml = pyamg.smoothed_aggregation_solver(A, B=B, BH=BH, strength=('symmetric', {'theta': 0.0}), smooth=('energy', {'krylov': 'cg', 'maxiter': 2, 'degree': 1, 'weighting': 'local'}), improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None], aggregate="standard",\
            presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),max_levels=15, max_coarse=300,coarse_solver="pinv")
            object.A=A
            print("construct solver")
            object.solver = ml
            print("solver constructed")
            object.xprev = B*0
            breakpoint()

        else:
            Osum,Os,Ow = assemble_Osum(object.H,object.tmask,object.dx,object.dy)
            object.Osum = Osum
            #object.Osum[:] = Osum[50,50]
            object.Os = Os
            object.Ow = Ow

    ##Osum=np.roll(Osum,1,axis=0)
    #Osum=np.roll(Osum,1,axis=1)
#)


    #beforeconv =  np.sum(np.abs(pi_rhs))
    if method=="mg":
        b = np.zeros(object.A.shape[0])
        b = assemble_mg_b(b,object.flatindexes,hu1+hu2,hv1+hv2,object.dx,object.dy,delt,object.umask,object.vmask,object.tmask)
        pi = object.RL[1]*0
        x = object.solver.solve(b,x0=object.xprev,tol=1e-12,maxiter=100)
        #print("residual: ",np.sum(np.abs(np.matmul(object.A,x)-b)))
        object.xprev=x
        for i in range(len(x)):
            pi[object.flatindexes==i]=x[i]
        #breakpoint()
        #pi[object.flatindexes>=0]=x
    else:
        pi_rhs = np.zeros(hu1.shape)
        pi_rhs = assemble_pi_rhs(pi_rhs,hu1+hu2,hv1+hv2,object.dx,object.dy,delt,object.umask,object.vmask)
        rp=0.66
        if object.pressure_solves >3:
            rp=0.66
        pi_tol = 10**-19
        pi = object.RL[1]
        iters = 0
        pi = SOR(pi,pi_rhs,object.Osum,object.Os,object.Ow,rp,pi_tol,pi.shape[0],pi.shape[1],object.tmask)
    #pi*0

    pi_x = -delt*(np.roll(pi,-1,axis=1)-pi)/(object.dx)*object.umask
    pi_x[np.roll(object.tmask,-1,axis=1)!=object.tmask]=0
    pi_y = -delt*(np.roll(pi,-1,axis=0)-pi)/(object.dy)*object.vmask
    pi_y[np.roll(object.tmask,-1,axis=0)!=object.tmask]=0
    
    if debug:
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax2.imshow(pi_x)
        ax3.imshow(pi_y)
        ax1.imshow(pi)
        plt.show()

    object.RL[1] = pi
    #pi=pi*object.tmask
    #plt.imshow(pi)
    #object.U[2,:,:] = object.Ustar + delt*pressure_diff_u(object,pi,1,1)/(object.dx)*object.umask
    #object.U2[2,:,:] = object.Ustar2 + delt*pressure_diff_u(object,pi,1,1)/(object.dx)*object.umask
    #object.V[2,:,:] = object.Vstar + delt*pressure_diff_v(object,pi,0,1)/(object.dy)*object.vmask
    #object.V2[2,:,:] = object.Vstar2 + delt*pressure_diff_v(object,pi,0,1)/(object.dy)*object.vmask
    object.U[2,:,:] = (object.Ustar + pi_x)*object.umask
    object.U2[2,:,:] = (object.Ustar2 + pi_x)*object.umask
    object.V[2,:,:] = (object.Vstar + pi_y)*object.vmask
    object.V2[2,:,:] = (object.Vstar2 + pi_y)*object.vmask

    #hu1 = object.U[2]*tools.ip_t(object,object.D[1])*object.umask
    #hv1 = object.V[2]*tools.jp_t(object,object.D[1])*object.vmask
    #hu2 = object.U2[2]*tools.ip_t(object,object.D2[1])*object.umask
    #hv2 = object.V2[2]*tools.jp_t(object,object.D2[1])*object.vmask

    #pi_rhs = np.zeros(hu1.shape)
    #pi_rhs = assemble_pi_rhs(pi_rhs,hu1+hu2,hv1+hv2,object.dx,object.dy,delt,object.umask,object.vmask,object.tmask)
    if debug:
        plt.imshow(pi)
        plt.show()
        plt.quiver(object.Ustar,object.Vstar)
        plt.show()
        plt.quiver(object.Ustar2,object.Vstar2)
        plt.show()
        plt.quiver(object.U[2],object.V[2])
        plt.show()
        plt.quiver(object.U2[2],object.V2[2])
        plt.show()
    #
    #breakpoint()

    object.pressure_solves+=1
    if object.pressure_solves == 7500:
        breakpoint()
    if (object.pressure_solves)%300 ==1 and debug:
        fig,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3)
        X,Y = np.meshgrid(range(object.nx+2)*object.dx,range(object.ny+2)*object.dy)
        im = ax1.pcolormesh(X,Y,object.D[2]*object.tmask)
        #ax1.quiver(object.U[2],object.V[2])
        #plt.colorbar(im,ax=ax1)
        im = ax2.pcolormesh(X,Y,object.U[2]*object.umask)
        plt.colorbar(im,ax=ax2)
        im = ax3.pcolormesh(X,Y,object.V[2]*object.vmask)
        plt.colorbar(im,ax=ax3)
##
        im = ax4.pcolormesh(X,Y,object.D2[2]*object.tmask)
        ##ax4.quiver(object.U2[2],object.V2[2])
        plt.colorbar(im,ax=ax4)
        im = ax5.pcolormesh(X,Y,object.U2[2]*object.umask)
        plt.colorbar(im,ax=ax5)
        im = ax6.pcolormesh(X,Y,object.V2[2]*object.vmask)
        plt.colorbar(im,ax=ax6)
#
        im = ax7.pcolormesh(X,Y,object.RL[1])
        plt.colorbar(im,ax=ax7)
        im = ax8.pcolormesh(X,Y,object.Ustar2)
        plt.colorbar(im,ax=ax8)
        im = ax9.pcolormesh(X,Y,object.Vstar2)
        plt.colorbar(im,ax=ax9)
        plt.show()
        breakpoint()

    if (object.pressure_solves)%50 ==0 or object.pressure_solves<5:
        print("-----")
        #print("before conv", beforeconv)
        print("after conv: ", print("residual: ",np.sum(np.abs(np.matmul(object.A,x)-b))))
        print("KE1: ",np.sqrt(np.sum((object.umask*object.U[2])**2 + (object.vmask*object.V[2])**2)))
        print("KE2: ",np.sqrt(np.sum((object.umask*object.U2[2])**2 + (object.vmask*object.V2[2])**2)))
        print("D1: ",np.sum(object.D[1])/np.sum(object.tmask))
        print("D2: ",np.sum(object.D2[1])/np.sum(object.tmask))
        print("D2min: ",np.min(object.D2[1][object.tmask==1]))
        print("D2max: ",np.max(object.D2[1][object.tmask==1]))
        print("totalvolume: ",np.sum(object.tmask*(object.D2[1]+object.D[1])))
        print("pressure sulves: ", object.pressure_solves)

    #plt.show()
    
    
    


    object.RL[1]=pi


def intT(object,delt):
    """Integrate T. Multipy RHS of dDT/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.T[2,:,:] = object.T[0,:,:] \
                    +div0((-object.T[1,:,:] * object.dDdt \
                    +  convT(object,object.D[1,:,:]*object.T[1,:,:]) \
                    +  object.nentr*object.Ta \
                    +  object.melt*object.Tb - object.gamT*(object.T[1,:,:]-object.Tb) \
                    +  object.Kh*lapT(object,object.T[0,:,:]) \
                    -  (object.T[0,:,:]-object.Ta)*object.conv2 \
                    ),object.D[1,:,:]) * object.tmask * delt

def intS(object,delt):
    """Integrate S. Multipy RHS of dDS/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.S[2,:,:] = object.S[0,:,:] \
                    +div0((-object.S[1,:,:] * object.dDdt \
                    +  convT(object,object.D[1,:,:]*object.S[1,:,:]) \
                    +  object.nentr*object.Sa \
                    +  object.Kh*lapT(object,object.S[0,:,:]) \
                    -  (object.S[0,:,:]-object.Sa)*object.conv2\
                    ),object.D[1,:,:]) * object.tmask * delt
