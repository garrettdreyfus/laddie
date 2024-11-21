import numpy as np
from physics import *
from tools import *
import pdb
import matplotlib.pyplot as plt
from numba import njit
import tools

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
    object.D[0][object.D[0]>object.H]=object.H[object.D[0]>object.H]
    object.D[1][object.D[1]>object.H]=object.H[object.D[1]>object.H]
    object.D[2][object.D[2]>object.H]=object.H[object.D[2]>object.H]
    object.D2 = (object.H-object.D)*object.tmask
    object.dDdt = (object.D[2,:,:]-object.D[0,:,:]) / (2*object.dt)
    object.dD2dt = (object.D2[2,:,:]-object.D2[0,:,:]) / (2*object.dt)
    object.Ddrho = object.D[1,:,:]*object.drho
    object.TWterm = object.g*(object.zb-object.D[1,:,:])*((object.drho))
    #object.TWterm = object.g*(object.zb-object.D[1,:,:])*(object.drho)
       
    if object.convop == 2:
        object.conv2 = np.where(object.drho<0,1,0)*object.D[1,:,:]/object.convtime# *np.where(object.convop==2,1,0)
    else:
        object.conv2 = 0

def timefilter(object):
    """Time filter, Robert Asselin scheme"""
    object.D[1,:,:] += object.nu/2 * (object.D[0,:,:]+object.D[2,:,:]-2*object.D[1,:,:]) * object.tmask
    object.U[1,:,:] += object.nu/2 * (object.U[0,:,:]+object.U[2,:,:]-2*object.U[1,:,:]) * object.umask
    object.V[1,:,:] += object.nu/2 * (object.V[0,:,:]+object.V[2,:,:]-2*object.V[1,:,:]) * object.vmask

    object.D2[1,:,:] = object.H-object.D[1]
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

    object.U2[2,:,:] = np.where(object.U[2,:,:]> object.vcut, object.vcut,object.U[2,:,:])
    object.U2[2,:,:] = np.where(object.U[2,:,:]<-object.vcut,-object.vcut,object.U[2,:,:])
    object.V2[2,:,:] = np.where(object.V[2,:,:]> object.vcut, object.vcut,object.V[2,:,:])
    object.V2[2,:,:] = np.where(object.V[2,:,:]<-object.vcut,-object.vcut,object.V[2,:,:])   

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

# def intU(object,delt):
#     """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
#     object.U[2,:,:] = object.U[0,:,:] \
#                     + div0((-object.U[1,:,:] * ip_t(object,object.dDdt) \
#                     +  convU(object) \
#                     #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

#                     ## PRESSURE TERMS
#                     ### --------------------
#                     -  .5*object.g*ip_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
#                     + ip_t(object,object.D[1,:,:])*(np.roll(object.RL,-1,axis=1)-object.RL)/(object.dx*object.rho0) \
#                     ### --------------------

#                     +  object.f*ip_t(object,object.D[1,:,:]*object.Vjm) \
#                     +  -object.Cd* object.U[1,:,:] *(object.U[1,:,:]**2 + ip(jm(object.V[1,:,:]))**2)**.5 \
#                     +  object.Ah*lapU(object) \
#                     +  -object.detr* object.U[1,:,:] \
#                     ),ip_t(object,object.D[1,:,:])) * object.umask * delt

# def intV(object,delt):
#     """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
#     object.V[2,:,:] = object.V[0,:,:] \
#                     +div0((-object.V[1,:,:] * jp_t(object,object.dDdt) \
#                     + convV(object) \
#                     #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     #PRESSURE TERMS
#                     #-------------------------
#                     - .5*object.g*jp_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     + jp_t(object,object.D[1,:,:])*(np.roll(object.RL,-1,axis=0)-object.RL)/(object.dy*object.rho0) \
#                     #-------------------------

#                     + -object.f*jp_t(object,object.D[1,:,:]*object.Uim) \
#                     + -object.Cd* object.V[1,:,:] *(object.V[1,:,:]**2 + jp(im(object.U[1,:,:]))**2)**.5 \
#                     + object.Ah*lapV(object) \
#                     +  -object.detr* object.V[1,:,:] \
#                     ),jp_t(object,object.D[1,:,:])) * object.vmask * delt

# def intU2(object,delt):
#     """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
#     object.U2[2,:,:] = object.U2[0,:,:] \
#                     + div0((-object.U2[1,:,:] * ip_t(object,object.dD2dt) \
#                     +  convU2(object) \
#                     #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

#                     ## PRESSURE TERMS
#                     ### --------------------
#                     #-  .5*object.g*ip_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D2[1,:,:]))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
#                     + ip_t(object,object.D2[1,:,:])*(np.roll(object.RL,-1,axis=1)-object.RL)/(object.dx*object.rho0) \
#                     - ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx*object.rho0)\
#                     ### --------------------

#                     +  object.f*ip_t(object,object.D2[1,:,:]*object.V2jm) \
#                     +  -object.Cd* object.U2[1,:,:] *(object.U2[1,:,:]**2 + ip(jm(object.V2[1,:,:]))**2)**.5 \
#                     +  object.Ah*lapU2(object) \
#                     +  -object.detr* object.U2[1,:,:] \
#                     ),ip_t(object,object.D2[1,:,:])) * object.umask * delt

# def intV2(object,delt):
#     """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
#     object.V2[2,:,:] = object.V2[0,:,:] \
#                     +div0((-object.V2[1,:,:] * jp_t(object,object.dD2dt) \
#                     + convV2(object) \
#                     #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     #PRESSURE TERMS
#                     #-------------------------
#                     #- .5*object.g*jp_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
#                     + jp_t(object,object.D2[1,:,:])*(np.roll(object.RL,-1,axis=0)-object.RL)/(object.dy*object.rho0) \
#                     - jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy*object.rho0)\
#                     #-------------------------

#                     + -object.f*jp_t(object,object.D2[1,:,:]*object.U2im) \
#                     + -object.Cd* object.V2[1,:,:] *(object.V2[1,:,:]**2 + jp(im(object.U2[1,:,:]))**2)**.5 \
#                     + object.Ah*lapV2(object) \
#                     +  -object.detr* object.V2[1,:,:] \
#                     ),jp_t(object,object.D2[1,:,:])) * object.vmask * delt

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
                    +  object.Ah*lapU(object) \
                    +  -object.detr* object.U[1,:,:] \
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
                    + object.Ah*lapV(object) \
                    +  -object.detr* object.V[1,:,:] \
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
                    +  -object.detr* object.U2[1,:,:] \
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
                    +  -object.detr* object.V2[1,:,:] \
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
def assemble_pi_rhs(pi_rhs,hu,hv,dx,dy,dt,umask,vmask):
    for i in range(1,pi_rhs.shape[0]):
        for j in range(1,pi_rhs.shape[1]):
            if vmask[i,j]!=0:
                pi_rhs[i][j] = pi_rhs[i,j] + hv[i,j]/(dy*dt)
                pi_rhs[i+1][j] = pi_rhs[i+1,j]- hv[i,j]/(dy*dt)
            if umask[i,j]!=0:
                pi_rhs[i][j] = pi_rhs[i,j] + hu[i,j]/(dx*dt)
                pi_rhs[i][j+1] = pi_rhs[i,j+1] - hu[i,j]/(dx*dt)
    return pi_rhs
#SOR_jit = jit(SOR)

#def create_pi_rhs(pi_rhs,hu1,hv1,hu2,hv2,dx,dy,vmask,umask):
    #for i in range:

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
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.imshow(Ow)
    ax2.imshow(Os)
    ax3.imshow((Os==0) & (tmask==1))
    ax4.imshow(tmask==1)
    plt.show()
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




def surface_pressure(object,delt):

    X,Y = np.meshgrid(range(object.nx+2)*object.dx,range(object.ny+2)*object.dy)
    X = (X-np.max(X)/2)/np.max(X)
    Y = (Y-np.max(Y)/2)/np.max(Y)
    Xnew = X/(0.1*(X**2+Y**2))
    Xnew[np.abs(Xnew)>100]=100*np.sign(Xnew[np.abs(Xnew)>100])
    Ynew = Y/(0.1*(X**2+Y**2))
    Ynew[np.abs(Ynew)>100]=100*np.sign(Ynew[np.abs(Ynew)>100])
    X=Xnew
    Y=Ynew
    print("-------")
    #object.vmask=np.roll(object.vmask,1,axis=0)
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
    object.D2[1][object.tmask==0]=0
    object.D[1][object.tmask==0]=0

    object.Vstar[:] = 0
    object.Ustar[:] = (-Y/10)*object.vmask#object.D[1]
    object.Vstar2[:] = 0
    object.Ustar2[:] = (-Y/10)*object.vmask#object.D2[1]

    #object.Vstar[:] = (-Y/10)*object.vmask#object.D[1]
    #object.Ustar[:] = 0
    #object.Vstar2[:] = (-Y/10)*object.vmask#object.D2[1]
    #object.Ustar2[:] = 0
    #hu1 = object.Ustar*Dw#im_t(object,object.D[1])
    #hv1 = object.Vstar*Ds#jm_t(object,object.D[1])
    #hu2 = object.Ustar2*D2s#im_t(object,object.D2[1])
    #hv2 = object.Vstar2*D2w#jm_t(object,object.D2[1])



    #You have to fix this !!!!
    hu1 = object.Ustar*10#tools.im_t(object,object.D[1])
    hv1 = object.Vstar*10#tools.jm_t(object,object.D[1])
    hu2 = object.Ustar2*10#tools.im_t(object,object.D2[1])
    hv2 = object.Vstar2*10#tools.jm_t(object,object.D2[1])
#

    if (object.pressure_solves)%1 ==0 and True:

        #termu = - object.g*ip_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=1)-object.drho)/(object.dx)
        #termu[object.umask==0]=np.nan
        #termv = - object.g*jp_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=0)-object.drho)/(object.dy) 
        #termv[object.vmask==0]=np.nan

        #termu2 = - ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx)
        #termu2[object.umask==0]=np.nan
        #termv2 = - jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy)
        #termv2[object.vmask==0]=np.nan
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        im = ax1.imshow(hu1)
        plt.colorbar(im,ax=ax1)
        im = ax2.imshow(hv1)
        plt.colorbar(im,ax=ax2)
        im = ax3.imshow(hu2)
        plt.colorbar(im,ax=ax3)
        im = ax4.imshow(hv2)
        plt.colorbar(im,ax=ax4)
        plt.show()




    pi_rhs = np.zeros(hu1.shape)
    pi_rhs = assemble_pi_rhs(pi_rhs,hu1+hu2,hv1+hv2,object.dx,object.dy,delt,object.umask,object.vmask)

    plt.imshow(object.grlW)
    plt.show()
    if object.pressure_solves==0:
        Osum,Os,Ow = assemble_Osum(object.H,object.tmask,object.dx,object.dy)
        object.Osum = Osum
        #object.Osum[:] = Osum[50,50]
        object.Os = Os
        object.Ow = Ow

    ##Osum=np.roll(Osum,1,axis=0)
    #Osum=np.roll(Osum,1,axis=1)
##
    plt.suptitle("OSUM")
    plt.imshow(object.Osum)
    plt.show()
    plt.suptitle("pi_rhs")
    #pi_rhs[object.tmask==0]=np.nan
    plt.imshow(pi_rhs)
    plt.show()
#

    print("conv: ", np.sum(np.abs(pi_rhs)))
    rp=0.66
    if object.pressure_solves >3:
        rp=1.5
    pi_tol = 10**-21
    pi = object.RL[1]
    iters = 0
    pi = SOR(pi,pi_rhs,object.Osum,object.Os,object.Ow,rp,pi_tol,pi.shape[0],pi.shape[1],object.tmask)
    plt.suptitle("pi")
    plt.imshow(pi)
    plt.show()



#
    #fig,(ax1,ax2) = plt.subplots(1,2)
    #pi_rhs[H==0] = np.nan
    #ax1.imshow(Osum)
    #ax2.imshow(H)
    #plt.show()

    object.RL[1] = pi
    #pi=pi*object.tmask
    #plt.imshow(pi)
    #object.U[2,:,:] = object.Ustar + delt*pressure_diff_u(object,pi,1,1)/(object.dx)*object.umask
    #object.U2[2,:,:] = object.Ustar2 + delt*pressure_diff_u(object,pi,1,1)/(object.dx)*object.umask
    #object.V[2,:,:] = object.Vstar + delt*pressure_diff_v(object,pi,0,1)/(object.dy)*object.vmask
    #object.V2[2,:,:] = object.Vstar2 + delt*pressure_diff_v(object,pi,0,1)/(object.dy)*object.vmask
    pi_x = -delt*(np.roll(pi,-1,axis=1)-pi)/(object.dx)
    pi_x[np.roll(object.tmask,-1,axis=1)!=object.tmask]=0
    pi_y = -delt*(np.roll(pi,-1,axis=0)-pi)/(object.dy)
    pi_y[np.roll(object.tmask,-1,axis=0)!=object.tmask]=0
    object.U[2,:,:] = (object.Ustar + pi_x)*object.umask
    object.U2[2,:,:] = (object.Ustar2 + pi_x)*object.umask
    object.V[2,:,:] = (object.Vstar + pi_y)*object.vmask
    object.V2[2,:,:] = (object.Vstar2 + pi_y)*object.vmask

    hu1 = object.U[2]*10#tools.im(object.D[1])
    hv1 = object.V[2]*10#tools.jm(object.D[1])
    hu2 = object.U2[2]*10#tools.im(object.D2[1])
    hv2 = object.V2[2]*10#tools.jm(object.D2[1])

    pi_rhs = np.zeros(hu1.shape)
    pi_rhs = assemble_pi_rhs(pi_rhs,hu1+hu2,hv1+hv2,object.dx,object.dy,delt,object.umask,object.vmask)
    print("conv: ", np.sum(np.abs(pi_rhs)))
    plt.imshow(pi_rhs)
    plt.show()
    plt.quiver(object.U[2],object.V[2])
    plt.show()
#
    breakpoint()

    object.pressure_solves+=1
    if (object.pressure_solves)%1 ==0:
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(pi_x)
        ax2.imshow(pi_y)
        plt.show()
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

    hu1 = object.Ustar*im_t(object,object.D[1])
    hv1 = object.Vstar*jm_t(object,object.D[1])
    hu2 = object.Ustar2*im_t(object,object.D2[1])
    hv2 = object.Vstar2*jm_t(object,object.D2[1])

    hu = hu1+hu2
    hv = hv1+hv2
    #ax1.imshow(object.tmask*(hu-np.roll(hu,-1,axis=1)+hv-np.roll(hv,-1,axis=0)))

    hu1 = object.U[2]*im_t(object,object.D[1])*object.umask
    hv1 = object.V[2]*jm_t(object,object.D[1])*object.vmask
    hu2 = object.U2[2]*im_t(object,object.D2[1])*object.umask
    hv2 = object.V2[2]*jm_t(object,object.D2[1])*object.vmask
    hu = hu1+hu2
    hv = hv1+hv2
    #ax2.imshow(object.tmask*((hu-np.roll(hu,-1,axis=1))/object.dx+(hv-np.roll(hv,-1,axis=0))/object.dy))
    print("KE: ",np.sqrt(np.sum((object.umask*object.U[2])**2 + (object.vmask*object.V[2])**2)))
    print("D1: ",np.sum(object.D[1])/np.sum(object.tmask))
    print("D2: ",np.sum(object.D2[1])/np.sum(object.tmask))
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
