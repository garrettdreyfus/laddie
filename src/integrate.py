import numpy as np
from physics import *
from tools import *
import pyamg
import pdb
import matplotlib.pyplot as plt

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
    object.dDdt = (object.D[2,:,:]-object.D[0,:,:]) / (2*object.dt)
    object.D2 = ((object.zb-object.B)-object.D)*object.tmask
    object.D2[object.D2<0]=0
    object.dD2dt = (object.D2[2,:,:]-object.D2[0,:,:]) / (2*object.dt)
    object.Ddrho = object.D[1,:,:]*object.drho
    object.TWterm = object.g*(object.zb-object.D[1,:,:])*((object.rho0-object.rho02)/object.rho0+object.drho)
       
    if object.convop == 2:
        object.conv2 = np.where(object.drho<0,1,0)*object.D[1,:,:]/object.convtime# *np.where(object.convop==2,1,0)
    else:
        object.conv2 = 0

def timefilter(object):
    """Time filter, Robert Asselin scheme"""
    object.D[1,:,:] += object.nu/2 * (object.D[0,:,:]+object.D[2,:,:]-2*object.D[1,:,:]) * object.tmask
    object.U[1,:,:] += object.nu/2 * (object.U[0,:,:]+object.U[2,:,:]-2*object.U[1,:,:]) * object.umask
    object.V[1,:,:] += object.nu/2 * (object.V[0,:,:]+object.V[2,:,:]-2*object.V[1,:,:]) * object.vmask

    object.D2[1,:,:] += object.nu/2 * (object.D2[0,:,:]+object.D2[2,:,:]-2*object.D2[1,:,:]) * object.tmask
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

def intU(object,delt):
    """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.U[2,:,:] = object.U[0,:,:] \
                    + div0((-object.U[1,:,:] * ip_t(object,object.dDdt) \
                    +  convU(object) \
                    #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

                    ## PRESSURE TERMS
                    ### --------------------
                    -  .5*object.g*ip_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
                    + ip_t(object,object.D[1,:,:])*(np.roll(object.RL,-1,axis=1)-object.RL)/(object.dx*object.rho0) \
                    ### --------------------

                    +  object.f*ip_t(object,object.D[1,:,:]*object.Vjm) \
                    +  -object.Cd* object.U[1,:,:] *(object.U[1,:,:]**2 + ip(jm(object.V[1,:,:]))**2)**.5 \
                    +  object.Ah*lapU(object) \
                    +  -object.detr* object.U[1,:,:] \
                    ),ip_t(object,object.D[1,:,:])) * object.umask * delt

def intV(object,delt):
    """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.V[2,:,:] = object.V[0,:,:] \
                    +div0((-object.V[1,:,:] * jp_t(object,object.dDdt) \
                    + convV(object) \
                    #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #PRESSURE TERMS
                    #-------------------------
                    - .5*object.g*jp_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    + jp_t(object,object.D[1,:,:])*(np.roll(object.RL,-1,axis=0)-object.RL)/(object.dy*object.rho0) \
                    #-------------------------

                    + -object.f*jp_t(object,object.D[1,:,:]*object.Uim) \
                    + -object.Cd* object.V[1,:,:] *(object.V[1,:,:]**2 + jp(im(object.U[1,:,:]))**2)**.5 \
                    + object.Ah*lapV(object) \
                    +  -object.detr* object.V[1,:,:] \
                    ),jp_t(object,object.D[1,:,:])) * object.vmask * delt

def intU2(object,delt):
    """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.U2[2,:,:] = object.U2[0,:,:] \
                    + div0((-object.U2[1,:,:] * ip_t(object,object.dD2dt) \
                    +  convU2(object) \
                    #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

                    ## PRESSURE TERMS
                    ### --------------------
                    #-  .5*object.g*ip_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D2[1,:,:]))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
                    + ip_t(object,object.D2[1,:,:])*(np.roll(object.RL,-1,axis=1)-object.RL)/(object.dx*object.rho0) \
                    - ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx*object.rho0)\
                    ### --------------------

                    +  object.f*ip_t(object,object.D2[1,:,:]*object.V2jm) \
                    +  -object.Cd* object.U2[1,:,:] *(object.U2[1,:,:]**2 + ip(jm(object.V2[1,:,:]))**2)**.5 \
                    +  object.Ah*lapU2(object) \
                    +  -object.detr* object.U2[1,:,:] \
                    ),ip_t(object,object.D2[1,:,:])) * object.umask * delt

def intV2(object,delt):
    """Integrate V. Multipy RHS of dDV/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.V2[2,:,:] = object.V2[0,:,:] \
                    +div0((-object.V2[1,:,:] * jp_t(object,object.dD2dt) \
                    + convV2(object) \
                    #+ object.g*jp_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #+ -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    #PRESSURE TERMS
                    #-------------------------
                    #- .5*object.g*jp_t(object,object.D2[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    + jp_t(object,object.D2[1,:,:])*(np.roll(object.RL,-1,axis=0)-object.RL)/(object.dy*object.rho0) \
                    - jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy*object.rho0)\
                    #-------------------------

                    + -object.f*jp_t(object,object.D2[1,:,:]*object.U2im) \
                    + -object.Cd* object.V2[1,:,:] *(object.V2[1,:,:]**2 + jp(im(object.U2[1,:,:]))**2)**.5 \
                    + object.Ah*lapV2(object) \
                    +  -object.detr* object.V2[1,:,:] \
                    ),jp_t(object,object.D2[1,:,:])) * object.vmask * delt

def generate_stars(object,delt):
    #hello
    """Integrate U. Multipy RHS of dDU/dt, divided by D, with delt (= 2x dt for LeapFrog)"""
    object.Ustar = object.U[0,:,:] \
                    + div0((-object.U[1,:,:] * ip_t(object,object.dDdt) \
                    +  convU(object) \
                    #+  -object.g*ip_t(object,object.D[1,:,:]*object.zb)*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \

                    ## PRESSURE TERMS
                    ### --------------------
                    -  .5*object.g*ip_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*aware_diff_t(object,object.drho,1,-1)/object.dx \
                            
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
                    - .5*object.g*jp_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
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
                    + object.g*ip_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=1)-object.TWterm)/(object.dx)\
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
                    + object.g*jp_t(object,object.D2[1,:,:])*(np.roll(object.TWterm,-1,axis=0)-object.TWterm)/(object.dy)\
                    #-------------------------

                    + -object.f*jp_t(object,object.D2[1,:,:]*object.U2im) \
                    + -object.Cd* object.V2[1,:,:] *(object.V2[1,:,:]**2 + jp(im(object.U2[1,:,:]))**2)**.5 \
                    + object.Ah*lapV2(object) \
                    +  -object.detr* object.V2[1,:,:] \
                    ),jp_t(object,object.D2[1,:,:])) * object.vmask * delt

def surface_pressure(object,delt):
    hu1 = object.Ustar*im_u(object,object.D[1])*object.umask
    hv1 = object.Vstar*jm_v(object,object.D[1])*object.vmask
    hu2 = object.Ustar2*im_u(object,object.D2[1])*object.umask
    hv2 = object.Vstar2*jm_v(object,object.D2[1])*object.vmask

    pi_rhs = np.zeros(hu1.shape)


    pi_rhs -= hv1/(object.dy*delt)
    pi_rhs += np.roll(hv1/(object.dy*delt),-1,axis=0)*object.vmask

    pi_rhs -= hv2/(object.dy*delt)
    pi_rhs += np.roll(hv2/(object.dy*delt),-1,axis=0)*object.vmask

    pi_rhs -= hu1/(object.dx*delt)
    pi_rhs += np.roll(hu1/(object.dx*delt),-1,axis=1)*object.umask

    pi_rhs -= hu2/(object.dx*delt)
    pi_rhs += np.roll(hu2/(object.dx*delt),-1,axis=1)*object.umask
    pi_rhs = pi_rhs*object.tmask
    #pi_rhs[object.tmask==0]=np.nan

    ## think thats good
    H = object.D[1]+object.D2[1]

    #plt.imshow((object.D2[1]<0))
    #plt.show()

    #Hs = (1/3) *  (H+np.roll(H,1,axis=0)+0.25*(np.roll(H,-1,axis=1)+np.roll(H,1,axis=1)+np.roll(np.roll(H,-1,axis=1),1,axis=0)+np.roll(np.roll(H,1,axis=1),1,axis=0)))
    preHs = np.roll(H,1,axis=0)
    preHs[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=0))]=H[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=0))]
    Hs = (H + preHs)/2

    preHw = np.roll(H,1,axis=1)
    preHw[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=1))]=H[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=1))]
    Hw = (H + preHw)/2
    #Hw = (1/3) *  (H+np.roll(H,1,axis=1)+0.25*(np.roll(H,-1,axis=0)+np.roll(H,1,axis=0)+np.roll(np.roll(H,-1,axis=0),1,axis=1)+np.roll(np.roll(H,1,axis=1),1,axis=0)))

    Os = Hs/(object.dy**2)*object.tmask
    Ow = Hw/(object.dx**2)*object.tmask

    rolledOw = np.roll(Ow,1,axis=1)
    rolledOw[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=1))]=Ow[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=1))]

    rolledOs = np.roll(Os,1,axis=0)
    rolledOs[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=0))]=Os[np.logical_and(object.tmask,~np.roll(object.tmask,1,axis=0))]

    Osum = Ow + rolledOw + Os + rolledOs 
    Osum=Osum*object.tmask
    #fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    #ax1.imshow(object.umask*-.5*object.g*ip_t(object,object.D[1,:,:]*(object.zb-object.D[1,:,:]/2))*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx)
    #ax1.imshow(object.vmask*aware_diff_u(object,object.drho,1,-1)/object.dx)
    #ax2.imshow(object.Ustar)
    tw = object.TWterm
    #ax3.imshow(H)
    #ax2.imshow(hu1+hu2)
    #ax3.imshow(hv1+hv2)
    Osum[Osum!=0] = 1/Osum[Osum!=0]
    zb = object.zb
    #ax4.imshow(object.vmask*.5*object.g*jp_t(object,object.D[1,:,:]*(object.zb+object.zb-object.D[1,:,:]))*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy)
    drho = object.drho
    #drho[object.tmask==0]=np.nan
    #ax4.imshow(object.drho)
    #plt.show()

    rp=0.75
    pi_tol = 10**-5
    pi_prev = object.RL[1]
    pi = object.RL[1]
    iters = 0
    plt.imshow(pi_rhs)
    plt.show()
    plt.imshow(pi)
    plt.show()
    while True:
        pi_prev[:] = pi
        pi = (1-rp)*pi \
                    + rp * Osum \
                        *  ( np.roll(Os*pi,-1,axis=0) + Os*np.roll(pi,1,axis=0) + np.roll(Ow*pi,-1,axis=1) + Ow*np.roll(pi,1,axis=1) - pi_rhs );
        pi = pi*object.tmask
        pi_prev = pi_prev*object.tmask

        maxdiff = np.nanmax(np.abs(pi-pi_prev))
        iters+=1
        if maxdiff<pi_tol:
            break
    print("SOR iters: ", iters)   
    object.RL[1][:] = pi

    object.U[2,:,:] = object.Ustar + delt*(np.roll(pi,1,axis=1)-pi)/(object.dx)*object.tmask
    object.U2[2,:,:] = object.Ustar2 + delt*(np.roll(pi,1,axis=1)-pi)/(object.dx)*object.tmask
    object.V[2,:,:] = object.Vstar + delt*(np.roll(pi,1,axis=0)-pi)/(object.dy)*object.tmask
    object.V2[2,:,:] = object.Vstar2 + delt*(np.roll(pi,1,axis=0)-pi)/(object.dy)*object.tmask
    object.pressure_solves+=1
    if (object.pressure_solves)%1000 ==0:
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.imshow(object.D[2])
        ax2.imshow(object.V[2])
        ax3.imshow(object.U[2])
        ax4.imshow(object.drho)
        #ax4.quiver(object.U2[2],object.V2[2])
        plt.show()
    fig,(ax1,ax2) = plt.subplots(1,2)
    hu1 = object.Ustar*im_u(object,object.D[1])*object.umask
    hv1 = object.Vstar*jm_v(object,object.D[1])*object.vmask
    hu2 = object.Ustar2*im_u(object,object.D2[1])*object.umask
    hv2 = object.Vstar2*jm_v(object,object.D2[1])*object.vmask
    hu = hu1+hu2
    hv = hv1+hv2
    print(np.nansum(object.umask*object.vmask*np.abs(hu-np.roll(hu,1,axis=1)+hv-np.roll(hv,1,axis=0))))
    ax1.imshow(object.umask*object.vmask*hu-np.roll(hu,1,axis=1)+hv-np.roll(hv,1,axis=0))

    hu1 = object.U[2]*im_u(object,object.D[1])*object.umask
    hv1 = object.V[2]*jm_v(object,object.D[1])*object.vmask
    hu2 = object.U2[2]*im_u(object,object.D2[1])*object.umask
    hv2 = object.V2[2]*jm_v(object,object.D2[1])*object.vmask
    hu = hu1+hu2
    hv = hv1+hv2
    print(np.nansum(object.umask*object.vmask*object.tmask*np.abs(hu-np.roll(hu,1,axis=1)+hv-np.roll(hv,1,axis=0))))
    print("-----")
    ax2.imshow(hu-np.roll(hu,1,axis=1)+hv-np.roll(hv,1,axis=0))
  
    plt.show()
    
    
    


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
