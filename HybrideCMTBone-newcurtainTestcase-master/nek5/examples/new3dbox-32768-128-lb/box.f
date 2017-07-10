      include "cmtparticles.usrp"
c-----------------------------------------------------------------------
c README -  JH111516 - Rarefaction test case. FIRST PLACE WE TRY
!           entropy viscosity in EUler gas dynamics! IF it works here,
!           fold it in!!
c README -  JH122716 - OK NOW TRY EVM after re-doing viscous stress tensor
!           and fixing some (not all. some) BC problems in Navier-Stokes
c-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'   ! this is not
      include 'CMTDATA' ! the best idea
      include 'NEKUSE'
      integer e,eg

      e = gllel(eg)

      i=ix+(iy-1)*ny1
!     artvisc=c_sub_e*gridh(i,e)**2*
!    >                   abs(res2(ix,iy,iz,e,1))/maxdiff
      artvisc=c_sub_e*meshh(e)**2*maxres(e)/maxdiff

      mu=0.0!rho*min(artvisc,t(ix,iy,iz,e,3))
c     mu=rho*min(artvisc,t(ix,iy,iz,e,3))
      nu_s=mu/rho

      mu=0.5*mu ! A factor of
           ! 2 lurks in agradu's evaluation of strain rate, even in EVM
      lambda=0.0
      udiff=0.0
      utrans=0.

      return
      end
c-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,eg

      common /part_two_way/  ptw
      real                   ptw(lx1,ly1,lz1,lelt,4) 
      e = gllel(eg)

c     ffx =  ptw(ix,iy,iz,e,1)/vtrans(ix,iy,iz,e,1) !Nek5000
c     ffy =  ptw(ix,iy,iz,e,2)/vtrans(ix,iy,iz,e,1)
c     ffz =  ptw(ix,iy,iz,e,3)/vtrans(ix,iy,iz,e,1)

      ffx =  ptw(ix,iy,iz,e,1) ! cmtnek
      ffy =  ptw(ix,iy,iz,e,2)
      ffz =  ptw(ix,iy,iz,e,3)

c     ffx = 0.0
c     ffy = 0.0
c     ffz = 0.0
      return
      end
c-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,eg

      qvol   = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'
      include 'TORO'
      include 'CMTDATA'
      integer  e,f

      real err(4),work(4)

      n = nx1*ny1*nz1*nelt
      ifxyo=.true.
      if (istep.gt.1) ifxyo=.false.

      eps     = 0.00001

      umin = glmin(t,n)
      umax = glmax(t,n)
!     if (mod(kstep,100).eq.0) then
      if (nio.eq.0) then
         write(6,2)istep,time_cmt,umin,' <T<',umax
      endif
!     endif
2     format(i6,1p2e17.8,a4,1p1e17.8)

      call rzero(err,4)
      do e=1,nelt
         do k=1,nz1
         do j=1,ny1
         do i=1,nx1
            s=(xm1(i,j,k,e)-diaph1)/(time+zerotime)
            CALL SAMPLE(PMstar, UM, S, rhohere, uhere, prshere)
            err(1)=err(1)+(u(i,j,k,1,e)-rhohere)**2*bm1(i,j,k,e)
            err(2)=err(2)+(u(i,j,k,2,e)-rhohere*uhere)**2*bm1(i,j,k,e)
            err(3)=err(3)+(u(i,j,k,3,e))**2*bm1(i,j,k,e)
            energy=prshere/(gmaref-1.0)+0.5*rhohere*uhere**2
            err(4)=err(4)+(u(i,j,k,5,e)-energy)**2*bm1(i,j,k,e)
         enddo
         enddo
         enddo
      enddo
      call gop(err,work,'+  ',4) ! Add across all processors
      call vsqrt(err,4)

      if(mod(istep,iostep).eq.0) then
! mus contains mu_s, nu_s, vz whatevs, mu_max, and entropy residual
         call outpost(vdiff(1,1,1,1,imu),vdiff(1,1,1,1,inus),
     >                vz,t(1,1,1,1,3),res2,'mus')
          if(nio.eq.0) write(37,'(i5,4e25.16)') istep,err
      endif ! mod(istep,iostep)
      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,eg)
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'TORO'
      include 'CMTDATA'
      include 'GEOM' ! not sure if this is a good idea.
      real nx,ny,nz  ! bite me it's fun
      integer e

!     e = gllel(eg)
      molarmass=molmass

      if (cbu .eq. 'W  ' .or. cbu .eq. 'I  ' .or. cbu .eq. 'SYM') then
         flux=0.0 ! not used in wall?
      elseif (cbu.eq.'O ') then
!        if (outflsub) pres=pinfty ! not yet. leave this in outflow_bc
      endif

      return
      end

c-----------------------------------------------------------------------

      subroutine useric (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'TORO'
      include 'PERFECTGAS'
      include 'CMTDATA'

! JH080614 actual arguments here (and corresponding dummy arguments)
!          are test3 ("left Woodward & Colella") in e1rpex.ini. They
!          will be appropriately distributed throughout the commons in
!          TORO along with the solution star state PMstar and UM.
! JH073114 Toro e1rpex provides SUBROUTINE SAMPLE and is crudely grafted
!          to the end of this .usr file.
        e=gllel(eg)

        s=(x-diaph1)/zerotime
        if (x.gt.diaph1) s=0.
        CALL SAMPLE(PMstar, UM, S, rho, ux, pres)
        uy = 0.
        uz = 0.
        phi = 1.0
        varsic(1) = phi*rho
        varsic(2) = varsic(1)*ux
        varsic(3) = varsic(1)*uy
        varsic(4) = varsic(1)*uz
        varsic(5) = varsic(1)*(cvgref*
     > MixtPerf_T_DPR(rho,pres,rgasref)+
     >  0.5*(ux**2+uy**2+uz**2))
        temp = MixtPerf_T_DPR(rho,pres,rgasref)
      cp=cpgref
      cv=cvgref
      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'CMTTIMERS'
      include 'CMTBCDATA'
      include 'PERFECTGAS'
c     molmass    = 8314.3
      molmass    = 29.
      muref      = 0.0
      coeflambda = -2.0/3.0
      suthcoef   = 1.0
      reftemp    = 1.0
      prlam      = 0.72
      gmaref     = 1.4
      rgasref    = MixtPerf_R_M(molmass,dum)
      cvgref     = rgasref/(gmaref-1.0)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 

      res_freq = 1000000
      flio_freq=iostep
      NCUT = 7!lx1 - int(lx1/2)
      NCUTD = 15!lxd
      wght  = 1.0
      wghtd = 1.0
      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2
      include 'SIZE'
      include 'TOTAL'
      include 'TORO'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'PERFECTGAS'   

      outflsub=.true.
      IFCNTFILT=.false.
      ifrestart=.false.
      ifsip=.false.
      gasmodel = 1
! JH080714 Now with parameter space to sweep through
      open(unit=81,file="riemann.inp",form="formatted")
      read (81,*) domlen
      read (81,*) xdiaph
      read (81,*) gmaref
      read (81,*) dleft
      read (81,*) uleft
      read (81,*) pleft
      read (81,*) dright
      read (81,*) uright
      read (81,*) pright
      read (81,*) zerotime
      close(81)

      molmass=8314.3
      muref=0.0
      coeflambda=-2.0/3.0
      suthcoef=1.0
      prlam = 0.72
      rgasref    = MixtPerf_R_M(molmass,dum)
      cvgref     = rgasref/(gmaref-1.0)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 

      c_max=0.5     ! should be 0.5, really
      c_sub_e=1.0e36

      call e1rpex(domlen,xdiaph,gmaref,dleft,uleft,pleft,dright,uright,
     >            pright,1.0)
      CALL SAMPLE(PMstar, UM, 0.0, rhohere, uhere, pinfty)
      reftemp=pleft/dleft/rgasref
      aleft=sqrt(gmaref*pleft/dleft)
      call domain_size(xmin,xmax,ymin,ymax,zmin,zmax)
      if(nio.eq.0)then
         write(6,*) 'domlen',domlen
         write(6,*) 'xdiaph',xdiaph
         write(6,*) 'gamma ',gmaref
         write(6,*) 'rhol  ',dleft
         write(6,*) 'ul    ',uleft
         write(6,*) 'pl    ',pleft
         write(6,*) 'rhor  ',dright
         write(6,*) 'ur    ',uright
         write(6,*) 'pr    ',pright
         write(6,*) 'sound ',aleft
         write(6,*) 'ustar ',um
         write(6,*) 'dt    ',dt
         write(6,*) 'nsteps',nsteps
         write(6,*) 'final time ',(xdiaph-xmin)/aleft
      endif
      return
      end
!-----------------------------------------------------------------------
      subroutine cmt_userEOS(ix,iy,iz,eg)
      include 'SIZE'
      include 'NEKUSE'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'PERFECTGAS'
      integer e,eg

      cp=cpgref
      cv=cvgref
      temp=e_internal/cv
      asnd=MixtPerf_C_GRT(gmaref,rgasref,temp)
      pres=MixtPerf_P_DRT(rho,rgasref,temp)
      return
      end
!-----------------------------------------------------------------------
      subroutine usrdat3
      return
      end
c-----------------------------------------------------------------------
      subroutine e1rpex(DOMin,DIAPHin,GAMMAin,DLin,ULin,PLin,DRin,URin,
     >                  PRin,PSCALEin)
*----------------------------------------------------------------------*
*                                                                      *
C     Exact Riemann Solver for the Time-Dependent                      *
C     One Dimensional Euler Equations                                  *
*                                                                      *
C     Name of program: HE-E1RPEX                                       *
*                                                                      *
C     Purpose: to solve the Riemann problem exactly,                   *
C              for the time dependent one dimensional                  *
C              Euler equations for an ideal gas                        *
*                                                                      *
C     Input  file: e1rpex.ini                                          *
C     Output file: e1rpex.out (exact solution)                         *
*                                                                      *
C     Programer: E. F. Toro                                            *
*                                                                      *
C     Last revision: 31st May 1999                                     *
*                                                                      *
C     Theory is found in Ref. 1, Chapt. 4 and in original              *
C     references therein                                               *
*                                                                      *
C     1. Toro, E. F., "Riemann Solvers and Numerical                   *
C                      Methods for Fluid Dynamics"                     *
C                      Springer-Verlag, 1997                           *
C                      Second Edition, 1999                            *
*                                                                      *
C     This program is part of                                          *
*                                                                      *
C     NUMERICA                                                         *
C     A Library of Source Codes for Teaching,                          *
C     Research and Applications,                                       *
C     by E. F. Toro                                                    *
C     Published by NUMERITEK LTD, 1999                                 *
C     Website: www.numeritek.com                                       *
*                                                                      *
*----------------------------------------------------------------------*
*
      include 'TORO'
*
C     Declaration of variables:
*
      INTEGER I, CELLS
*
*
C     Input variables
*
C     DOMLEN   : Domain length
C     DIAPH1   : Position of diaphragm 1
C     CELLS    : Number of computing cells
C     GAMMA    : Ratio of specific heats
C     TIMEOU   : Output time
C     DL       : Initial density  on left state
C     UL       : Initial velocity on left state
C     PL       : Initial pressure on left state
C     DR       : Initial density  on right state
C     UR       : Initial velocity on right state
C     PR       : Initial pressure on right state
C     PSCALE   : Normalising constant
*
!     Initial data and parameters are now arguments

           DOMLEN=DOMin
           DIAPH1=DIAPHin
           GAMMA =GAMMAin
           DL    =DLin
           UL    =ULin
           PL    =PLin
           DR    =DRin
           UR    =URin
           PRight=PRin
           PSCALE=PSCALEin

C     Compute gamma related constants
*
      G1 = (GAMMA - 1.0)/(2.0*GAMMA)
      G2 = (GAMMA + 1.0)/(2.0*GAMMA)
      G3 = 2.0*GAMMA/(GAMMA - 1.0)
      G4 = 2.0/(GAMMA - 1.0)
      G5 = 2.0/(GAMMA + 1.0)
      G6 = (GAMMA - 1.0)/(GAMMA + 1.0)
      G7 = (GAMMA - 1.0)/2.0
      G8 = GAMMA - 1.0
*
C     Compute sound speeds
*
      CL = SQRT(GAMMA*PL/DL)
      CR = SQRT(GAMMA*PRight/DR)
*
C     The pressure positivity condition is tested for
*
      IF(G4*(CL+CR).LE.(UR-UL))THEN
*
C        The initial data is such that vacuum is generated.
C        Program stopped.
*
         WRITE(6,*)
         WRITE(6,*)'***Vacuum is generated by data***'
         WRITE(6,*)'***Program stopped***'
         WRITE(6,*)
*
         call exitt
      ENDIF
*
C     Exact solution for pressure and velocity in star
C     region is found
*
      CALL STARPU(PMstar, UM, PSCALE)
*
      return
      end
*
*----------------------------------------------------------------------*
*
      SUBROUTINE STARPU(P, U, PSCALE)
*
c     IMPLICIT NONE
*
C     Purpose: to compute the solution for pressure and
C              velocity in the Star Region
*
C     Declaration of variables
*
      INTEGER I, NRITER
*
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        CHANGE, FL, FLD, FR, FRD, P, POLD, PSTART,
     &        TOLPRE, U, UDIFF, PSCALE
*
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR
      DATA TOLPRE, NRITER/1.0E-06, 20/
*
C     Guessed value PSTART is computed
*
      CALL GUESSP(PSTART)
*
      POLD  = PSTART
      UDIFF = UR - UL
*
c     WRITE(6,*)'----------------------------------------'
c     WRITE(6,*)'   Iteration number      Change  '
c     WRITE(6,*)'----------------------------------------'
*
      DO 10 I = 1, NRITER
*
         CALL PREFUN(FL, FLD, POLD, DL, PL, CL)
         CALL PREFUN(FR, FRD, POLD, DR, PRight, CR)
         P      = POLD - (FL + FR + UDIFF)/(FLD + FRD)
         CHANGE = 2.0*ABS((P - POLD)/(P + POLD))
c        WRITE(6, 30)I, CHANGE
         IF(CHANGE.LE.TOLPRE)GOTO 20
         IF(P.LT.0.0)P = TOLPRE
         POLD  = P
*
 10   CONTINUE
*
      WRITE(6,*)'Divergence in Newton-Raphson iteration'
*
 20   CONTINUE
*
C     Compute velocity in Star Region
*
      U = 0.5*(UL + UR + FR - FL)
*
c     WRITE(6,*)'---------------------------------------'
c     WRITE(6,*)'   Pressure        Velocity'
c     WRITE(6,*)'---------------------------------------'
c     WRITE(6,40)P/PSCALE, U
c     WRITE(6,*)'---------------------------------------'
*
 30   FORMAT(5X, I5,15X, F12.7)
 40   FORMAT(2(F14.6, 5X))
*
      END
*
*----------------------------------------------------------------------*
*
      SUBROUTINE GUESSP(PMstar)
*
C     Purpose: to provide a guessed value for pressure
C              PM in the Star Region. The choice is made
C              according to adaptive Riemann solver using
C              the PVRS, TRRS and TSRS approximate
C              Riemann solvers. See Sect. 9.5 of Chapt. 9
C              of Ref. 1
*
c     IMPLICIT NONE
*
C     Declaration of variables
*
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8,
     &        CUP, GEL, GER, PMstar, PMAX, PMIN, PPV, PQ,
     &        PTL, PTR, QMAX, QUSER, UM
*
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR
*
      QUSER = 2.0
*
C     Compute guess pressure from PVRS Riemann solver
*
      CUP  = 0.25*(DL + DR)*(CL + CR)
      PPV  = 0.5*(PL + PRight) + 0.5*(UL - UR)*CUP
      PPV  = MAX(0.0, PPV)
      PMIN = MIN(PL,  PRight)
      PMAX = MAX(PL,  PRight)
      QMAX = PMAX/PMIN
*
      IF(QMAX.LE.QUSER.AND.
     & (PMIN.LE.PPV.AND.PPV.LE.PMAX))THEN
*
C        Select PVRS Riemann solver
*
         PMstar = PPV
      ELSE
         IF(PPV.LT.PMIN)THEN
*
C           Select Two-Rarefaction Riemann solver
*
            PQ  = (PL/PRight)**G1
            UM  = (PQ*UL/CL + UR/CR +
     &            G4*(PQ - 1.0))/(PQ/CL + 1.0/CR)
            PTL = 1.0 + G7*(UL - UM)/CL
            PTR = 1.0 + G7*(UM - UR)/CR
            PMstar  = 0.5*(PL*PTL**G3 + PRight*PTR**G3)
         ELSE
*
C           Select Two-Shock Riemann solver with
C           PVRS as estimate
*
            GEL = SQRT((G5/DL)/(G6*PL + PPV))
            GER = SQRT((G5/DR)/(G6*PRight + PPV))
            PMstar=(GEL*PL+GER*PRight-(UR-UL))/(GEL+GER)
         ENDIF
      ENDIF
*
      END
*
*----------------------------------------------------------------------*
*
      SUBROUTINE PREFUN(F,FD,P,DK,PK,CK)
*
C     Purpose: to evaluate the pressure functions
C              FL and FR in exact Riemann solver
C              and their first derivatives
*
c     IMPLICIT NONE
*
C     Declaration of variables
*
      REAL    AK, BK, CK, DK, F, FD, P, PK, PRATIO, QRT,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
*
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
*
      IF(P.LE.PK)THEN
*
C        Rarefaction wave
*
         PRATIO = P/PK
         F    = G4*CK*(PRATIO**G1 - 1.0)
         FD   = (1.0/(DK*CK))*PRATIO**(-G2)
      ELSE
*
C        Shock wave
*
         AK  = G5/DK
         BK  = G6*PK
         QRT = SQRT(AK/(BK + P))
         F   = (P - PK)*QRT
         FD  = (1.0 - 0.5*(P - PK)/(BK + P))*QRT
      ENDIF
*
      END
*
*----------------------------------------------------------------------*
*
      SUBROUTINE SAMPLE(PMstar, UM, S, D, U, P)
*
C     Purpose: to sample the solution throughout the wave
C              pattern. Pressure PM and velocity UM in the
C              Star Region are known. Sampling is performed
C              in terms of the 'speed' S = X/T. Sampled
C              values are D, U, P
*
C     Input variables : PMstar, UM, S, /GAMMAS/, /STATES/
C     Output variables: D, U, P
*
c     IMPLICIT NONE
*
C     Declaration of variables
*
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8,
     &        C, CML, CMR, D, P, PMstar, PML, PMR,  S,
     &        SHL, SHR, SL, SR, STL, STR, U, UM
*
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR

      IF(S.LE.UM)THEN
*
C        Sampling point lies to the left of the contact
C        discontinuity
*
         IF(PMstar.LE.PL)THEN
*
C           Left rarefaction
*
            SHL = UL - CL
*
            IF(S.LE.SHL)THEN
*
C              Sampled point is left data state
*
               D = DL
               U = UL
               P = PL
            ELSE
               CML = CL*(PMstar/PL)**G1
               STL = UM - CML
*
               IF(S.GT.STL)THEN
*
C                 Sampled point is Star Left state
*
                  D = DL*(PMstar/PL)**(1.0/GAMMA)
                  U = UM
                  P = PMstar
               ELSE
*
C                 Sampled point is inside left fan
*
                  U = G5*(CL + G7*UL + S)
                  C = G5*(CL + G7*(UL - S))
                  D = DL*(C/CL)**G4
                  P = PL*(C/CL)**G3
               ENDIF
            ENDIF
         ELSE
*
C           Left shock
*
            PML = PMstar/PL
            SL  = UL - CL*SQRT(G2*PML + G1)
*
            IF(S.LE.SL)THEN
*
C              Sampled point is left data state
*
               D = DL
               U = UL
               P = PL
*
            ELSE
*
C              Sampled point is Star Left state
*
               D = DL*(PML + G6)/(PML*G6 + 1.0)
               U = UM
               P = PMstar
            ENDIF
         ENDIF
      ELSE
*
C        Sampling point lies to the right of the contact
C        discontinuity
*
         IF(PMstar.GT.PRight)THEN
*
C           Right shock
*
            PMR = PMstar/PRight
            SR  = UR + CR*SQRT(G2*PMR + G1)
*
            IF(S.GE.SR)THEN
*
C              Sampled point is right data state
*
               D = DR
               U = UR
               P = PRight
            ELSE
*
C              Sampled point is Star Right state
*
               D = DR*(PMR + G6)/(PMR*G6 + 1.0)
               U = UM
               P = PMstar
            ENDIF
         ELSE
*
C           Right rarefaction
*
            SHR = UR + CR
*
            IF(S.GE.SHR)THEN
*
C              Sampled point is right data state
*
               D = DR
               U = UR
               P = PRight
            ELSE
               CMR = CR*(PMstar/PRight)**G1
               STR = UM + CMR
*
               IF(S.LE.STR)THEN
*
C                 Sampled point is Star Right state
*
                  D = DR*(PMstar/PRight)**(1.0/GAMMA)
                  U = UM
                  P = PMstar
               ELSE
*
C                 Sampled point is inside left fan
*
                  U = G5*(-CR + G7*UR + S)
                  C = G5*(CR - G7*(UR - S))
                  D = DR*(C/CR)**G4
                  P = PRight*(C/CR)**G3
               ENDIF
            ENDIF
         ENDIF
      ENDIF
*
      END
*
c----------------------------------------------------------------------c
*
      subroutine cmt_usrflt(rmult)
      include 'SIZE'
      real rmult(lx1)
      real alpfilt
      integer sfilt, kut
      real eta, etac
      call rone(rmult,lx1)
      alpfilt=36.0 ! H&W 5.3
      kut=lx1/2
      sfilt=8
      etac=real(kut)/real(nx1)
      do i=kut,nx1
         eta=real(i)/real(nx1)
         rmult(i)=exp(-alpfilt*((eta-etac)/(1.0-etac))**sfilt)
      enddo
      return
      end

c----------------------------------------------------------------------c


      subroutine compute_entropy(s)
! computes entropy at istep and pushes the stack down for previous
! steps needed to compute ds/dt via finite difference (for now).
! hardcoded for Burgers equation. More later when folded into CMT-nek
! for Burgers, s=energy=1/2 U^2
      include 'SIZE'
      include 'TOTAL'  ! tlag is lurking. be careful
      include 'CMTDATA'
! I've always seen lorder=3, but I still need three steps
!          s(:,               1,       1)  ! entropy at current step
!          s(:,               2,       1)  ! entropy at step n-1
!          s(:,               1,       2)  ! entropy at step n-2
      real s(lx1*ly1*lz1*lelt,lorder-1,*)
      real ntol
      integer e
      data icalld /0/
      save icalld

      n=nx1*ny1*nz1
      ntot=n*nelt
      ntol=1.0e-10

      if (icalld .eq. 0) then
         icalld=1
         call rzero(s,ntot)
         call rzero(s(1,1,2),ntot)
         call rzero(s(1,2,1),ntot)
      endif
! push the stack
      call copy(s(1,1,2),s(1,2,1),ntot) ! s_{n-2}=s_{n-1}
      call copy(s(1,2,1),s(1,1,1),ntot) ! s_{n-1}=s_n

! compute the current entropy
      rgam=1.0/(gmaref-1.0)
      do i=1,ntot
         rho=max(vtrans(i,1,1,1,irho),ntol)
         s(i,1,1)=rgam*rho*log(pr(i,1,1,1)/(rho**gmaref))
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine entropy_viscosity
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ scrent(lxyz,lelt)
      real visco(lxyz,nelt)
      integer e

      pi=4.0*atan(1.0)

      goto 999
      n=nx1*ny1*nz1
      ntot=n*nelt
! entropy at this and lorder prior steps
      call compute_entropy(tlag)
! compute maxval(|S-<S>|)
      savg    =    glsc2(tlag,bm1,ntot)
      savg    = -savg/volvm1
      call cadd2(scrent,tlag,savg,ntot)
      maxdiff =     glamax(scrent,ntot)
      call entropy_residual(tlag) ! into res2
      call wavevisc(t(1,1,1,1,3))
      call resvisc(res2) ! fill maxres for now
!     write(6,*) 'max |s-<s>|=',maxdiff
999   continue
      return
      end

c-----------------------------------------------------------------------

      subroutine entropy_residual(s)
! COMPUTE R=ds/dt + div F(s) for entropy pair s and (hardcoded) F
! and store its norm functional thing |R| in res2 where it will
! provide artificial viscosity according to the code in
! entropy_viscosity and the method of Guermond
      include 'SIZE'
      include 'TOTAL' ! tlag lurks
      include 'CMTDATA'
      common /ctmp0/ restmp(lx1*ly1*lz1*lelt)
      real restmp
      real s(lx1*ly1*lz1*lelt,lorder-1,ldimt)
      integer e

      n=nx1*ny1*nz1
      ntot=n*nelt

! I set istep=kstep. why do I need to do this again?
      if (istep .eq. 1) return
! ds/dt via 1-sided finite difference. CHANGE THIS WHEN WE FINALLY
! HAVE VARIABLE DT
      r2dt=0.5/DT
      if (istep .eq. 2) then ! don't do this learn how to start multistep!!!
         rdt=2.0*r2dt
         call sub3(res2,s(1,1,1),s(1,2,1),ntot)
         call cmult(res2,rdt,ntot)
      elseif (istep .gt. 2) then
         call add3s2(res2,s(1,1,1),s(1,1,2),3.0,1.0,ntot)
         call add2s2(res2,s(1,2,1),-4.0,ntot)
         call cmult(res2,r2dt,ntot)
      endif
      call copy(t(1,1,1,1,2),res2,ntot) ! Bala suggested just using this
! res2=ds/dt. now,

!-----------------------------------------------------------------------
! cons approach: strong-conservation form of flux divergence in entropy residual
!-----------------------------------------------------------------------
! get around to expanding totalh to store fluxes for whole fields and
! properly vectorize evaluate_*_h and flux_div_integral
      do e=1,nelt
!! diagnostic
!         if (mod(istep,iostep).eq.0) then ! compare ds/dt to resid
!            do i=1,n
!               write(nid*1000+istep,*) xm1(i,1,1,e),ym1(i,1,1,e),
!     >                            res2(i,1,1,e,1)
!            enddo
!         endif
!! diagnostic
         call evaluate_entropy_flux(e) ! diffh. zero it before diffusion 
         call flux_div_mini(e) ! into res2 it goes.
      enddo
      maxres=glamax(res2,ntot)
! actually...
!-----------------------------------------------------------------------
! conv1 approach: convective form of flux divergence in entropy residual
!-----------------------------------------------------------------------
! Guermond Pasquetti & Popov (2011) (2.26) suggest a pointwise convective
! form for the entropy residual. see cburg.usr for further exploration

      return
      end

c-----------------------------------------------------------------------

      subroutine evaluate_entropy_flux(e)
! entropy flux function for entropy residual.
! just vel*s for now
      include 'SIZE'
      include 'SOLN'
      include 'INPUT'
      include 'CMTDATA'
      integer e

      call rzero(totalh,3*nxd*nyd*nzd)
      n=nx1*ny1*nz1

      call col3(totalh(1,1),vx(1,1,1,e),tlag(1,1,1,e,1,1),n)
      call col3(totalh(1,2),vy(1,1,1,e),tlag(1,1,1,e,1,1),n)
      if (if3d) call col3(totalh(1,3),vz(1,1,1,e),tlag(1,1,1,e,1,1),n)

      return
      end

c-----------------------------------------------------------------------

      subroutine flux_div_mini(e)
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'DXYZ'
      include 'SOLN'
      include 'CMTDATA'
      parameter (ldd=lx1*ly1*lz1)
      parameter (ldg=lx1**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju

      integer e

      nrstd=ldd
      nxyz=nx1*ny1*nz1
      mdm1=nx1-1

      call rzero(ur,nrstd)
      call rzero(us,nrstd)
      call rzero(ut,nrstd)
      call rzero(ud,nrstd)
      call rzero(tu,nrstd)

      if (if3d) then
         call local_grad3(ur,us,ut,totalh(1,1),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)
     $                        + sxm1(i,1,1,e)*us(i)
     $                        + txm1(i,1,1,e)*ut(i))
         enddo
         call local_grad3(ur,us,ut,totalh(1,2),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz ! confirmed to have no effect in 1D
            ud(i)=ud(i)+jacmi(i,e)*(rym1(i,1,1,e)*ur(i)
     $                            + sym1(i,1,1,e)*us(i)
     $                            + tym1(i,1,1,e)*ut(i))
         enddo
         call local_grad3(ur,us,ut,totalh(1,3),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz ! confirmed to have no effect in 1D
            ud(i)=ud(i)+jacmi(i,e)*(rzm1(i,1,1,e)*ur(i)
     $                            + szm1(i,1,1,e)*us(i)
     $                            + tzm1(i,1,1,e)*ut(i))
         enddo
      else
         call local_grad2(ur,us,totalh(1,1),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i) = jacmi(i,e)*(rxm1(i,1,1,e)*ur(i)
     $                        + sxm1(i,1,1,e)*us(i))
         enddo
         call local_grad2(ur,us,totalh(1,2),mdm1,1,dxm1,dxtm1)
         do i=1,nxyz
            ud(i)=ud(i)+jacmi(i,e)*(rym1(i,1,1,e)*ur(i)
     $                            + sym1(i,1,1,e)*us(i))
         enddo
      endif
      call add2(res2(1,1,1,e,1),ud,nxyz)

      return
      end

!-----------------------------------------------------------------------

      subroutine resvisc(residual)
! max entropy visc, defined by Guermond, Popov, whoever (chapter-verse)
! as 
! in a given element
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'DG'
      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt,
     >                   heresize=nqq*3*lfq,
     >                   hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      real flux,graduf
      parameter (lxyz=lx1*ly1*lz1)
      real residual(lxyz,nelt)
      integer e,f
      real maxjump

      nxyz =nx1*ny1*nz1
      nxz  =nx1*nz1
      nface=2*ndim
      nxzf =nxz*nface
      nfq  =nxzf*nelt

      iqm=1
      iqp=iqm+4*nfq

      call fillq(1,tlag,flux(iqm),flux(iqp))
      call fillq(2,vx,flux(iqm),flux(iqp))
      call fillq(3,vy,flux(iqm),flux(iqp))
      call fillq(4,vz,flux(iqm),flux(iqp))

      call jumpflux(flux(iqm),flux(iqp))

      l=1
      do e=1,nelt
         maxres(e)=vlamax(residual(1,e),nxyz)
         maxjump=vlamax(flux(l),nxzf)/meshh(e) ! |s u.n|/h
         l=l+nxzf
!        if (maxjump .gt. maxres(e)) maxres(e)=maxjump
      enddo

      return
      end

      subroutine jumpflux(qminus,jump)
      include 'SIZE'
      include 'GEOM'
      include 'DG'
      include 'INPUT'
      include 'CMTDATA'
      real qminus(nx1*nz1,2*ndim,nelt,4),
     >       jump(nx1*nz1,2*ndim,nelt)
      integer e,f
      character*3 cb

      nxz =nx1*nz1
      nface=2*ndim
      nxzf=nface*nxz
      nfq=nxzf*nelt

! form s u.n !!!!RFFFFFFFUUUUUUUUUUU unx is fixed at 6
!     call col3   (jump,unx,qminus(1,1,1,iux),nfq)
!     call addcol3(jump,uny,qminus(1,1,1,iuy),nfq)
!     if (if3d) call addcol3(jump,unz,qminus(1,1,1,iuz),nfq)
!     call col2(jump,qminus,nfq) ! 1 assumed to be entropy

! BC check. DO NOTHING FOR NOW
      ifield=1 ! FIX THIS

      do e=1,nelt
         do f=1,2*ndim
            cb=cbc(f,e,ifield)
            if (cb.ne.'E  '.and.cb.ne.'P  ') then
               call rzero(jump(1,f,e),nxz)
            else
               call col3   (jump(1,f,e),unx(1,1,f,e),qminus(1,f,e,iux),
     >                      nxz)
               call addcol3(jump(1,f,e),uny(1,1,f,e),qminus(1,f,e,iuy),
     >                      nxz)
               if (if3d)
     >         call addcol3(jump(1,f,e),unz(1,1,f,e),qminus(1,f,e,iuz),
     >                      nxz)
               call col2(jump(1,f,e),qminus(1,f,e,1),nxz) ! 1 assumed to be entropy
            endif
         enddo
      enddo

      call gs_op(dg_hndl,jump,1,1,0)

      return
      end

!-----------------------------------------------------------------------

      subroutine wavevisc(numax)
! max entropy visc, defined by Guermond, Popov, whoever (chapter-verse)
! as 
! numax = c_max*h*max(dH/dU)
! in a given element
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      parameter (lxyz=lx1*ly1*lz1)
      common /scrns/ wavespeed(lxyz)
      real wavespeed
      real maxeig
      real numax(lxyz,nelt)
      integer e

      nxyz=nx1*ny1*nz1

      do e=1,nelt
         do i=1,nxyz
            wavespeed(i)=csound(i,1,1,e)+
     >      sqrt(vx(i,1,1,e)**2+vy(i,1,1,e)**2+vz(i,1,1,e)**2)
         enddo
         maxeig=vlamax(wavespeed,nxyz)
         rhomax(e)=vlamax(vtrans(1,1,1,e,irho),nxyz)
         do i=1,nxyz
!           smax(i,e)=c_max*maxeig*gridh(i,e)
!           smax(i,e)=c_max*maxeig*meshh(e)*rhomax(e)
            numax(i,e)=c_max*maxeig*meshh(e)
         enddo
      enddo

      return
      end
c
c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)
      return
      end
c
c automatically added by makenek
      subroutine cmt_switch ! to set IFCMT logical flag
      include 'SIZE'
      include 'INPUT'
      IFCMT=.true.
      return
      end
c
c automatically added by makenek
      subroutine userflux ! user defined flux
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      real fluxout(lx1*lz1)
      return
      end
