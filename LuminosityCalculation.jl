#
#
#   LuminosityCalculation.jl
#   A script to calculate LHC luminosity & pileup conditions
#   Author          : Nikos Karastathis (nkarast .at. cern .dot. ch)
#   Version         : 0.1
#   Compatible with : Julia 0.6
#


using Roots
using Dierckx
using Cubature
using NLsolve

#
#   N.B. Using Cubature in infinite domains needs a transformation from -Inf, Inf to [-1, 1]. This requires a change of variables:
#   \int_{-\infty}^{\infty} f(x)dx = \int_{-1}^{1} f( t/(1-t^2) ) * (1+t^2)/(1-t^2)^2
#
#   x ---> t/(1-t^2)
#   So for two variables:
#
#   \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x,y)dxdy = \int_{-1}^{1} \int_{-1}^{1} f( a/(1-a^2) , b/(1-b^2)  ) * ((1+a^2)/(1-a^2)^2) * ((1+b^2)/(1-b^2)^2) da db
#
#   Source : https://github.com/stevengj/cubature#Infinite_intervals
#
#


const Clight                 = 299792458.                # speed of light [m/s]
const Qe                     = 1.60217733e-19            # electron charge [C]
const sigprotoninelas        = 0.081                     # inelastic hadron cross section [barn]
const sigtotproton           = 0.111                     # inelastic hadron cross section [barn]
const Mproton                = 0.93827231                # proton mass [GeV]

####################################################################################################
#
#   CURRENT SETUP PARAMETERS : NOMINAL SCENARIO AT 5e34
#                              SNAPSHOT AT THE BEGGINING AND END OF THE COAST
#
####################################################################################################

bx                = 0.15
by                = 0.15
dx                = 0 # always = 0
dy                = 0
nb                = 2736
Npart             = 2.2e11
const Nrj         = 7000. #GeV
const gamma_rel   = Nrj/Mproton

enX               = 2.5e-06
enY               = 2.5e-06

emitX             = enX/gamma_rel   # r.m.s. horizontal physical emittance in collision
emitY             = enY/gamma_rel   # r.m.s. vertical physical emittance in collision

const circum      = 26658.8832
const sigz        = 0.081
const frev        = Clight/circum
const hrf400      = 35640.
const omegaCC     = hrf400*frev/Clight*2.*pi

const VRF_ref     = 11.4                   # reference CC voltage for full crabbing at 590 murad crossing angle
const VRFx_ref    = 6.8 # 0. # LHC no CC   # CC voltage [MV] in crossing plane for 2 CCs
const VRFy_ref    = 0.0                    # default CC voltage [MV] in parallel plane

const alpha_ref   = 380.e-06               # Default full crossing angle
alpha             = 380.e-06  # full crossing now

const nIP         = 2

####################################################################################################
####################################################################################################
####################################################################################################

#
#   KERNEL FUNCTION
#
function kernel(z,t)
    VRFx     = min(VRFx_ref, alpha/alpha_ref*VRF_ref)
    VRFy     = 0
    omegaCCx = omegaCC
    omegaCCy = omegaCC
    return 1/sqrt(1 + (z/bx)^2)/sqrt(1 + (z/by)^2)*exp(-((dx*sqrt(bx*emitX) + alpha*z - alpha_ref/omegaCCx/VRF_ref*(VRFx*cos(omegaCCx*t)*sin(omegaCCx*z)))/(2*sqrt(bx*emitX))/sqrt(1 + (z/bx)^2))^2)*exp(-((dy*sqrt(by*emitY) + VRFy/VRF_ref*alpha_ref/omegaCCy*sin(omegaCCy*t)*cos(omegaCCy*z))/(2*sqrt(by*emitY))/sqrt(1 + (z/by)^2))^2)
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   DISTRIBUTION FUNCTION
#
function rho(z)
    return 1./sqrt(2.*pi)/sigz*exp(-(z*z)/2/(sigz*sigz))
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   DENSITY FUNCTION
#
function density(z, t) #x[1]=z, x[2]=t
    return 2*kernel(z,t)*rho(z-t)*rho(z+t)
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


function density_rloss_integrand(x)
    z = x[1]
    t = x[2]

    znew = z/(1-z^2)
    tnew = t/(1-t^2)

    return density(znew,tnew)*( (1+z^2)/(1-z^2)^2)*( (1+t^2)/(1-t^2)^2)
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

#
#   GENERALIZED LOSS FACTOR
#
function Rloss()
    # calculate Rloss:
    #val, err = si.dblquad(density, -Inf, Inf, x->-Inf, y->Inf)   # I'm using Python since Julia integtation on infinite domains requires me to study Calculus I...
    val, err = hcubature(density_rloss_integrand, [-1,-1], [1,1])  # performing a weird change of variables : https://github.com/stevengj/cubature#Infinite_intervals
    return val #, err
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
# LUMINOSITY [e34 Hz/cm^2]
#
function lumi()
    return (Rloss()*1./4./pi/sqrt(bx*by)/sqrt(emitX*emitY)*frev*nb*Npart^2)*1.0e-38
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *



#
#   TOTAL PILEUP FOR GIVEN CONFIGURATION
#
function mutot()
    return (lumi()*1.0e38*sigprotoninelas*1.0e-28)/(nb*frev)
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   NORMALIZED LINE PILEUP DENSITY [evt/mm] vs z[m]
#
function muz()
    integrand(t) = density(0, t/(1-t^2))*( (1+t^2)/(1-t^2)^2)
    return mutot()/Rloss()*hquadrature(integrand, -1, 1)[1]*1.0e-3
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   RMS LUMINOUS REGION [cm]
#
function siglumz_integrand(x)
    z = x[1]
    t = x[2]

    znew = z/(1-z^2)
    tnew = t/(1-t^2)

    return density(znew,tnew)*( (1+z^2)/(1-z^2)^2)*( (1+t^2)/(1-t^2)^2)*znew*znew
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

function siglumz()
    return 100.*sqrt(hcubature(siglumz_integrand, [-1, -1], [1,1])[1]/Rloss())
end
# # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *



#
#   FIND BETA VALUE FOR TARGET LUMI
#
function betaForLumi(targetLumi)
    function loc_lumi_func(mbeta)
        global bx
        global by
        bx = mbeta
        by = mbeta
        return lumi() - targetLumi
    end

    return fzero(loc_lumi_func, 0.6);#-0.1, 1.0, 1.0);
    # return secant_method(loc_lumi_func, -0.1, 1.1)
    #return nlsolve(loc_lumi_func!, -1.)
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET ENERGY [GeV]
#
function setNrj(GeV)
    Nrj = GeV;
    gamma_rel = Nrj/pmass;
    emitX = enX/gamma_rel;
    emitY = enY/gamma_rel;
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

#
#   SET BUNCH INTENSITY [ppb]
#
function setNrj(ppb)
    Npart = ppb;
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

#
#   SET HORIZONTAL NORMALIZED EMITTANCE [m]
#
function setenx(um)
    enX = um;
    emitX = enX / gamma_rel;
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET VERTICAL NORMALIZED EMITTANCE [m]
#
function seteny(um)
    enY = um;
    emitY = enY / gamma_rel;
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

#
#   SET  BETA* X [m]   
#
function setbx(m)
    bx = m
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET  BETA* Y [m]   
#
function setby(m)
    by = m
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET  SEPARATION X [sigma]   
#
function setsepx(sigma)
    dx = sigma
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET SEPARATION Y [sigma]   
#
function setsepx(sigma)
    dx = sigma
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *


#
#   SET CROSSING ANGLE [rad]   
#
function setAlpha(rad)
    alpha = rad
end
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

#
#   PRINT OUT BASIC STUFF
#
function printResult()
    println(" --- CALCULATING LUMINOSITY FOR THE CONDITIONS BELOW ---")
    println("N1, N2      = $Npart [ppb]")
    println("Nrj         = $Nrj [GeV]")
    println("Nb          = $nb [#]")
    println("beta* x     = $bx [m]")
    println("beta* y     = $by [m]")
    println("sep x       = $dx [sigma]")
    println("sep y       = $dy [sigma]")
    println("alpha       = $alpha [rad]")
    println("Norm. EmitX = $enX [m rad]")
    println("Norm. EmitY = $enY [m rad]")
    println("-----------------------------------")
    my_lumi = lumi()
    my_mutot = mutot()
    my_muz   = muz()
    my_lumz = siglumz()

    println("Luminosity      [e34 Hz/cm^2] : $my_lumi")
    println("Total Pileup    [events]      : $my_mutot")
    println("Peak  Pileup    [events/mm]   : $my_muz")
    println("Luminous Region [cm]          : $my_lumz")
end


# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
#           WORK IN PROGRESS
#
#
# const m_target_lumi      = 5.0 #e38;     # Hz/m/m
# const min_beta           = 0.15;       # m
# const fill_time          = 1*3600;    # sec

# ist_L              = m_target_lumi;
# int_L              = 0;

# const extra_losses       = 1.0; # multiplicative factor
# const time_step          = 60;  # integration step seconds
# const time_const_beta    = 60;  # beta* is changed every x seconds
# const print_step         = 60;  # output the data every x seconds

# const levelling = true

# const intensities = [0.1, 0.8, 1.0, 1.15, 1.275, 1.4, 1.6, 1.9, 2.2, 10 ]
# const xings       = [424., 424., 450.,  476.,   428., 414., 404., 350., 312., 312.] # 6 sigma

# const xing_I = Spline1D(intensities, xings)

# for t in 0:time_step:fill_time
#     alpha = xing_I(Npart*1.0e-11)*1.0e-6;

#     if levelling
#         if t % time_const_beta == 0
#             betaForLumi(m_target_lumi)
#             println(bx)
#             if bx < min_beta
#                 levelling = false # end of levelling
#                 bx = min_beta
#                 by = min_beta
#                 ist_L = lumi()
#             end
#         else
#             ist_L = lumi()
#         end
#         println(lumi())
#     end


#     # printing
#     if t % print_step == 0
#         if t==0
#              println("# 1. t [h]")
#              println("# 2. beta [m]")
#              println("# 3. Xing [urad]")
#              println("# 4. int_L [pb^1]")
#              println("# 5. ist_L [1e34 Hz/cm^2]")
#              println("# 6. total pileup]")
#              println("# 7. r.m.s. luminous region [cm]")
#              println("# 8. peak pileup [evt/m]")
#              println("# 9. Intensity [10^{11} p]")
#         end #t==0
#         @printf "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" t/3600. bx xing_I(Npart*1.0e-11) int_L*1.0e-40 ist_L*1.0e-38 mutot() siglumz()*100. muz() Npart*1.0e-11

# #         if t%time_step*20 ==0
# #             t_temp = Int(t/(fill_time/3600)/36)
# #             println("progress : $t_temp")
# #         end
#     end # t % print step


#     ist_L = lumi()*1.0e38;

#     int_L = int_L + ist_L*time_step;
#     dN = ist_L*sigprotoninelas*1.0e-28*time_step/nb*extra_losses*nIP;
#     Npart = Npart - dN;
# end # for
################################################################################