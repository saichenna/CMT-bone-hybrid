      SUBROUTINE AUSM_FluxFunction(ntot,nx,ny,nz,nm,fs,rl,ul,vl,wl,pl,
     >                         al,tl,rr,ur,vr,wr,pr,ar,tr,flx,cpl,cpr)

!     IMPLICIT NONE ! HAHAHHAHHAHA
! ******************************************************************************
! Definitions and declarations
! ******************************************************************************
      real MixtPerf_Ho_CpTUVW
      external MixtPerf_Ho_CpTUVW

! ==============================================================================
! Arguments
! ==============================================================================
      integer ntot
      REAL al(ntot),ar(ntot),fs(ntot),nm(ntot),nx(ntot),ny(ntot),
     >     nz(ntot),pl(ntot),pr(ntot),rl(ntot),rr(ntot),ul(ntot),
     >     ur(ntot),vl(ntot),vr(ntot),wl(ntot),wr(ntot),cpl(ntot),
     >     cpr(ntot),tl(ntot),tr(ntot)! INTENT(IN) ::
      REAL flx(ntot,5)!,vf(3) ! INTENT(OUT) ::

! ==============================================================================
! Locals
! ==============================================================================
      REAL af,mf,mfa,mfm,mfp,ml,mla,mlp,mr,mra,mrm,pf,ql,qr,vml,vmr,
     >        wtl,wtr,Hl,Hr

! ******************************************************************************
! Start, compute face state
! ******************************************************************************
      call invcol2(cpl,rl,ntot)
      call invcol2(cpr,rr,ntot)

      do i=1,ntot
         Hl = MixtPerf_Ho_CpTUVW(cpl(i),tl(i),ul(i),vl(i),wl(i))
         Hr = MixtPerf_Ho_CpTUVW(cpr(i),tr(i),ur(i),vr(i),wr(i))

         ql = ul(i)*nx(i) + vl(i)*ny(i) + wl(i)*nz(i) - fs(i)
         qr = ur(i)*nx(i) + vr(i)*ny(i) + wr(i)*nz(i) - fs(i)

         af = 0.5*(al(i)+ar(i)) ! NOTE not using original formulation, see note
         ml  = ql/af
         mla = ABS(ml)

         mr  = qr/af
         mra = ABS(mr)

         IF ( mla .le. 1.0 ) THEN
            mlp = 0.25*(ml+1.0)*(ml+1.0) + 0.125*(ml*ml-1.0)*(ml*ml-1.0)
            wtl = 0.25*(ml+1.0)*(ml+1.0)*(2.0-ml) +
     >            0.1875*ml*(ml*ml-1.0)*(ml*ml-1.0)
         ELSE
            mlp = 0.5*(ml+mla)
            wtl = 0.5*(1.0+ml/mla)
         END IF ! mla

         IF ( mra .le. 1.0 ) THEN
            mrm = -0.25*(mr-1.0)*(mr-1.0)-0.125*(mr*mr-1.0)*(mr*mr-1.0)
            wtr = 0.25*(mr-1.0)*(mr-1.0)*(2.0+mr) -
     >            0.1875*mr*(mr*mr-1.0)*(mr*mr-1.0)
         ELSE
            mrm = 0.5*(mr-mra)
            wtr = 0.5*(1.0-mr/mra)
         END IF ! mla

         mf  = mlp + mrm
         mfa = ABS(mf)
         mfp = 0.5*(mf+mfa)
         mfm = 0.5*(mf-mfa)

         pf = wtl*pl(i) + wtr*pr(i)

! ******************************************************************************
! Compute fluxes
! ******************************************************************************

!        vf(1) = mfp*ul + mfm*ur ! I'm sure we'll need this someday
!        vf(2) = mfp*vl + mfm*vr
!        vf(3) = mfp*wl + mfm*wr

         flx(i,1)=(af*(mfp*rl(i)      +mfm*rr(i)   )        )*nm(i)
         flx(i,2)=(af*(mfp*rl(i)*ul(i)+mfm*rr(i)*ur(i))+pf*nx(i))*
     >            nm(i)
         flx(i,3)=(af*(mfp*rl(i)*vl(i)+mfm*rr(i)*vr(i))+pf*ny(i))*
     >            nm(i)
         flx(i,4)=(af*(mfp*rl(i)*wl(i)+mfm*rr(i)*wr(i))+pf*nz(i))*
     >            nm(i)
         flx(i,5)=(af*(mfp*rl(i)*Hl   +mfm*rr(i)*Hr) + pf*fs(i))*
     >            nm(i)
      enddo

      return
      END

!-----------------------------------------------------------------------

      SUBROUTINE CentralInviscid_FluxFunction(ntot,nx,ny,nz,fs,ul,pl,
     >                                     ur,pr,flx)
! JH081915 More general, more obvious
! JH111815 HEY GENIUS THIS MAY BE SECOND ORDER AND THUS KILLING YOUR
!          CONVERGENCE. REPLACE WITH AUSM AND SHITCAN IT
! JH112015 This isn't why walls aren't converging. There's something
!          inherently second-order about your wall pressure. Think!
      real nx(ntot),ny(ntot),nz(ntot),fs(ntot),ul(ntot,5),pl(ntot),
     >     ur(ntot,5),pr(ntot) ! intent(in)
      real flx(ntot,5)! intent(out),dimension(5) ::

      do i=1,ntot
         rl =ul(i,1)
         rul=ul(i,2)
         rvl=ul(i,3)
         rwl=ul(i,4)
         rel=ul(i,5)

         rr =ur(i,1)
         rur=ur(i,2)
         rvr=ur(i,3)
         rwr=ur(i,4)
         rer=ur(i,5)

         ql = (rul*nx(i) + rvl*ny(i) + rwl*nz(i))/rl - fs(i)
         qr = (rur*nx(i) + rvr*ny(i) + rwr*nz(i))/rr - fs(i)

         flx(i,1) = 0.5*(ql* rl+ qr*rr               )
         flx(i,2) = 0.5*(ql* rul+pl(i)*nx(i) + qr* rur     +pr(i)*nx(i))
         flx(i,3) = 0.5*(ql* rvl+pl(i)*ny(i) + qr* rvr     +pr(i)*ny(i))
         flx(i,4) = 0.5*(ql* rwl+pl(i)*nz(i) + qr* rwr     +pr(i)*nz(i))
         flx(i,5) = 0.5*(ql*(rel+pl(i))+pl(i)*fs(i)+qr*(rer+pr(i))+
     >               pr(i)*fs(i))
      enddo

      return
      end

c----------------------------------------------------------------------------------
      subroutine nekadd2col2(a,b,c,n)
      real a(1),b(1),c(1)
c
      do i=1,n
         a(i) = a(i) + b(i)*c(i)
      enddo
      return
      end


c----------------------------------------------------------------------------------
      subroutine neklocal_grad3(ur,us,ut,u,N,e,D,Dt)
c     Output: ur,us,ut         Input:u,N,e,D,Dt
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real u (0:N,0:N,0:N,1)
      real D (0:N,0:N),Dt(0:N,0:N)
      integer e
c
      m1 = N+1
      m2 = m1*m1
c
      call nekmxm(D ,m1,u(0,0,0,e),m1,ur,m2) !- already implemented by Jason
      do k=0,N
         call nekmxm(u(0,0,k,e),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call nekmxm(u(0,0,0,e),m2,Dt,m1,ut,m1)
c
      return
      end

c----------------------------------------------------------------------------------
      subroutine neklocal_grad2(ur,us,u,N,e,D,Dt)
c     Output: ur,us         Input:u,N,e,D,Dt
      real ur(0:N,0:N),us(0:N,0:N)
      real u (0:N,0:N,1)
      real D (0:N,0:N),Dt(0:N,0:N)
      integer e
c
      m1 = N+1
c
      call nekmxm(D ,m1,u(0,0,e),m1,ur,m1)
      call nekmxm(u(0,0,e),m1,Dt,m1,us,m1)
c
      return
      end
c----------------------------------------------------------------------------------

      subroutine izero8(a,n)
      integer*8 a(1)
      do i=1,n
         a(i)=0
      enddo
      return
      end
!-----------------------------------------------------------------------

      subroutine nekmap_faced(ju,u,ba,ab,fdim,idir) ! GLL->GL interpolation

c     GLL interpolation from mx to md for a face array of size (nx,nz)

c     If idir ^= 0, then apply transpose operator  (md to mx)

      include 'SIZE'

      real    ju(1),u(1)
      integer fdim

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      real ba(ldg),ab(ldg)

      parameter (ld=2*lxd)
      common /ctmp0/ w(ld**ldim,2)

      if (idir.eq.0) then
         if (fdim.eq.2) then
            call mxm(ba,md,u,mx,wkd,mx)
            call mxm(wkd,md,ab,mx,ju,md)
         else
            call mxm(ba,md,u,mx,ju,1)
         endif
      else
         if (fdim.eq.2) then
            call mxm(ab,mx,u,md,wkd,md)
            call mxm(wkd,mx,ba,md,ju,mx)
         else
            call mxm(ab,mx,u,md,ju,1)
         endif
      endif

      return
      end

