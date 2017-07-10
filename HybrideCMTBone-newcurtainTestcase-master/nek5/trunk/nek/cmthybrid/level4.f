      subroutine nekinvcol3(a,b,c,n)
      REAL A(1),B(1),C(1)
C
      include 'OPCTR'
      include 'CTIMER'

#ifndef NOTIMER
      if (icalld.eq.0) tinv3=0.0
      icalld=icalld+1
      ninv3=icalld
      etime1=dnekclock()
C
C
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'invcl3'
      endif
      isbcnt = n
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      DO 100 I=1,N
         A(I)=B(I)/C(I)
 100  CONTINUE
#ifndef NOTIMER
      tinv3=tinv3+(dnekclock()-etime1)
#endif
      return
      END


c----------------------------------------------------------------------------------
      subroutine faceu(ivar,yourface)
! get faces of conserved variables stored contiguously
      include 'SIZE'
      include 'CMTDATA'
      include 'DG'
      integer e
      real yourface(nx1,nz1,2*ldim,nelt)

      do e=1,nelt
         call full2face_cmt(1,nx1,ny1,nz1,iface_flux(1,e),
     >                      yourface(1,1,1,e),u(1,1,1,ivar,e))
      enddo

      return
      end


c----------------------------------------------------------------------------------
      subroutine face_state_commo(mine,yours,nf,nstate,handle)

! JH060414 if we ever want to be more intelligent about who gets what,
!          who gives what and who does what, this is the place where all
!          that is done. At the very least, gs_op may need the transpose
!          flag set to 1. Who knows. Everybody duplicates everything for
!          now.
! JH070714 figure out gs_op_fields, many, vec, whatever (and the
!          corresponding setup) to get this done for the transposed
!          ordering of state variables. I want variable innermost, not
!          grid point.

      integer handle,nf,nstate ! intent(in)
      real yours(*),mine(*)

      ntot=nf*nstate
!      call nekcopy(yours,mine,ntot) !- level2.txt
!-----------------------------------------------------------------------
! operation flag is second-to-last arg, an integer
!                                                1 ==> +
!     write(6,*) 'face_state_commo ',nstate
      !print *,'in face state commo'
      gs_time = dnekclock()
      call gs_op_fields(handle,yours,nf,nstate,1,1,0)
      !print *,'gs time::',dnekclock()-gs_time,nf,nstate
!      call neksub2 (yours,mine,ntot) !- level4.txt
      return
      end

c----------------------------------------------------------------------------------
      subroutine nektranspose(a,lda,b,ldb)
      real a(lda,1),b(ldb,1)
c
      do j=1,ldb
         do i=1,lda
            a(i,j) = b(j,i)
         enddo
      enddo
      return
      end

c----------------------------------------------------------------------------------
c----------------------------------------------------------------------------------
      subroutine nekfacind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
c      ifcase in preprocessor notation
       KX1=1
       KY1=1
       KZ1=1
       KX2=NX
       KY2=NY
       KZ2=NZ
       IF (IFACE.EQ.1) KY2=1
       IF (IFACE.EQ.2) KX1=NX
       IF (IFACE.EQ.3) KY1=NY
       IF (IFACE.EQ.4) KX2=1
       IF (IFACE.EQ.5) KZ2=1
       IF (IFACE.EQ.6) KZ1=NZ
      return
      end


c----------------------------------------------------------------------------------
      subroutine add_face2full_cmt(nel,nx,ny,nz,iface,vols,faces)

      include 'SIZE'
      include 'TOTAL'

      integer   nel,nx,ny,nz
      integer   iface(nx*nz*2*ldim,*)
      real     faces(nx*nz   ,2*ldim,*  )
      real     vols (nx,ny,nz       ,nel)
      integer  ie,i,j

      n= nx*nz*2*ndim
      do ie=1,nel
      do j=1,n
         i=iface(j,ie)
         vols(i,1,1,ie) = vols(i,1,1,ie)+faces(j,1,ie)
      enddo
      enddo

      return
      end

c----------------------------------------------------------------------------------
      subroutine evaluate_conv_h(e,eq,ba,ab)
      include  'SIZE'
      include  'SOLN'
      include  'DEALIAS'
      include  'CMTDATA'
      include  'INPUT'

      parameter (ldd=lxd*lyd*lzd, ldg=lxd**3)
      common /ctmp1/ ju1(ldd),ju2(ldd)!,ur(ldd),us(ldd),ud(ldd),tu(ldd)
      common /dgradl/ jgl(ldg),jgt(ldg)
      real jgl,jgt

      real ju1,ju2
      integer  e,eq

C      print *, jgl(1)

! we add the convective fluxes, pressure and other terms
      n = nxd*nyd*nzd

      if (eq .eq. 1) then ! convective flux of mass=rho u_j=U_{j+1}

         do j=1,ndim
            !call nekintp_rstd(convh(1,j),u(1,1,1,eq+j,e),ba,ab,if3d,0)
            call intp_rstd(convh(1,j),u(1,1,1,eq+j,e),nx1,nxd,if3d,0)
         enddo

      else

         !call nekintp_rstd(ju1,phig(1,1,1,e),ba,ab,if3d,0)
         !call nekintp_rstd(ju2,pr(1,1,1,e),ba,ab,if3d,0)
         call intp_rstd(ju1,phig(1,1,1,e),nx1,nxd,if3d,0)
         call intp_rstd(ju2,pr(1,1,1,e),nx1,nxd,if3d,0)

         if (eq .lt. 5) then ! self-advection of rho u_i by rho u_i u_j

            !call nekintp_rstd(convh(1,1),u(1,1,1,eq,e),ba,ab,if3d,0)
            call intp_rstd(convh(1,1),u(1,1,1,eq,e),nx1,nxd,if3d,0)
            do j=2,ndim
               call nekcopy(convh(1,j),convh(1,1),n)
            enddo
            call nekcol2(convh(1,1),vxd(1,1,1,e),n)
            call nekcol2(convh(1,2),vyd(1,1,1,e),n)
            if (if3d) call col2(convh(1,3),vzd(1,1,1,e),n)
            call nekadd2col2(convh(1,eq-1),ju1,ju2,n)

         elseif (eq .eq. 5) then

            !call nekintp_rstd(convh(1,1),u(1,1,1,eq,e),ba,ab,if3d,0)
            call intp_rstd(convh(1,1),u(1,1,1,eq,e),nx1,nxd,if3d,0)
            call nekadd2col2(convh(1,1),ju1,ju2,n)
            do j=2,ndim
               call nekcopy(convh(1,j),convh(1,1),n)
            enddo
            call nekcol2(convh(1,1),vxd(1,1,1,e),n)
            call nekcol2(convh(1,2),vyd(1,1,1,e),n)
            call nekcol2(convh(1,3),vzd(1,1,1,e),n)

         else
            if(nio.eq.0) write(6,*) 'eq=',eq,'really must be <= 5'
            if(nio.eq.0) write(6,*) 'aborting in evaluate_conv_h'
            call exitt
         endif

      endif

      return
      end
c----------------------------------------------------------------------------------
      subroutine add_conv_diff_h
      include  'SIZE'
      include  'CMTDATA'

      n = nxd*nyd*nzd*3

!BLAS is also a possibility here

!     call add3(totalh,convh,diffh,n)
      call nekadd2(totalh,convh,n)

      return
      end

c----------------------------------------------------------------------------------
      subroutine nekgradl_rst(ur,us,ut,u,md,if3d)  ! GLL-based gradient
c
      include 'SIZE'
      include 'DXYZ'

      real    ur(1),us(1),ut(1),u(1)
      logical if3d

c     dgradl holds GLL-based derivative / interpolation operators

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgradl/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      m0 = md-1
      if (if3d) then
         call neklocal_grad3(ur,us,ut,u,m0,1,d(ip),dt(ip)) !- level5.txt
      else
         call neklocal_grad2(ur,us   ,u,m0,1,d(ip),dt(ip)) !- level5.txt
      endif

      return
      end

c----------------------------------------------------------------------------------
      subroutine nekcol2(a,b,n)
      real a(1),b(1)
      include 'OPCTR'

#ifndef NOTIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'col2  '
      endif
      isbcnt = N
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)*b(i)
      enddo

      return
      end

c----------------------------------------------------------------------------------
      subroutine nekadd2(a,b,n)
      real a(1),b(1)
      include 'OPCTR'

#ifndef NOTIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'ADD2  '
      endif
      isbcnt = N
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)+b(i)
      enddo
      return
      end

c----------------------------------------------------------------------------------
      subroutine full2face_cmt(nel,nx,ny,nz,iface,faces,vols)

! JH062314 Store face data from nel full elements (volume data). Merely
!          selection for the time being (GLL grid), but if we need to
!          extrapolate to faces (say, from Gauss points), this is where
!          we'd do it.

      include 'SIZE'
      include 'TOTAL'

      integer  nel,nx,ny,nz
      integer  iface(nx*nz*2*ldim,1)
      real     faces(nx*nz   ,2*ldim,nel)
      real     vols (nx,ny,nz       ,1  )
      integer  e,i,j

      n= nx*nz*2*ndim
      do e=1,nel
      do j=1,n
         i=iface(j,e)
         faces(j,1,e) = vols(i,1,1,e)
      enddo
      enddo

      return
      end

c----------------------------------------------------------------------------------
      subroutine neksub2(a,b,n)
      REAL A(1),B(1)
C
      include 'OPCTR'
C
#ifndef NOTIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'sub2  '
      endif
      isbcnt = n
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      DO 100 I=1,N
         A(I)=A(I)-B(I)
 100  CONTINUE
      return
      END

c----------------------------------------------------------------------------------
      subroutine neksubcol3(a,b,c,n)
      REAL A(1),B(1),C(1)
C
      include 'OPCTR'
C
#ifndef NOTIMER
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'subcl3'
      endif
      isbcnt = 2*n
      dct(myrout) = dct(myrout) + (isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + (isbcnt)
#endif
C
      DO 100 I=1,N
         A(I)=A(I)-B(I)*C(I)
  100 CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine iface_vert_int8cmt(nx,ny,nz,fa,va,jz0,jz1,nel)
      include 'SIZE'
      integer*8 fa(nx*nz,2*ndim,nel),va(0:nx+1,0:ny+1,jz0:jz1,nel)
      integer e,f

      n = nx*nz*2*ndim*nel
      call izero8(fa,n)

      mx1 = nx+2
      my1 = ny+2
      mz1 = nz+2
      if (ndim.eq.2) mz1=1

      nface = 2*ndim
      do e=1,nel
      do f=1,nface
         call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,f)

         if     (f.eq.1) then ! EB notation
            ky1=ky1-1
            ky2=ky1
         elseif (f.eq.2) then
            kx1=kx1+1
            kx2=kx1
         elseif (f.eq.3) then
            ky1=ky1+1
            ky2=ky1
         elseif (f.eq.4) then
            kx1=kx1-1
            kx2=kx1
         elseif (f.eq.5) then
            kz1=kz1-1
            kz2=kz1
         elseif (f.eq.6) then
            kz1=kz1+1
            kz2=kz1
         endif

         i = 0
         do iz=kz1,kz2
         do iy=ky1,ky2
         do ix=kx1,kx2
            i = i+1
            fa(i,f,e)=va(ix,iy,iz,e)
         enddo
         enddo
         enddo
      enddo
      enddo

      return
      end

      subroutine nekintp_rstd(ju,u,ba,ab,if3d,idir) ! GLL->GL interpolation

c     GLL interpolation from mx to md.

c     If idir ^= 0, then apply transpose operator  (md to mx)

      include 'SIZE'

      real    ju(1),u(1)
      logical if3d

      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      parameter (ld=2*lxd)
      common /ctmp0/ w(ld**ldim,2)

      ldw = 2*(ld**ldim)

      !call get_int_ptr (i,mx,md)
c
      if (idir.eq.0) then
         call specmpn(ju,md,u,mx,ba,ab,if3d,w,ldw)
      else
         call specmpn(ju,mx,u,md,ab,ba,if3d,w,ldw)
      endif
c
      return
      end

      
      subroutine InviscidFlux(qminus,qplus,flux,nstate,nflux)
!-------------------------------------------------------------------------------
! JH091514 A fading copy of RFLU_ModAUSM.F90 from RocFlu
!-------------------------------------------------------------------------------

!#ifdef SPEC
!      USE ModSpecies, ONLY: t_spec_type
!#endif
      include 'SIZE'
      include 'INPUT' ! do we need this?
      include 'GEOM' ! for unx
      include 'CMTDATA' ! do we need this without outflsub?
      include 'DG'

! ==============================================================================
! Arguments
! ==============================================================================
      integer nstate,nflux
      real qminus(nx1*nz1,2*ndim,nelt,nstate),
     >     qplus(nx1*nz1,2*ndim,nelt,nstate),
     >     flux(nx1*nz1,2*ndim,nelt,nflux)

! ==============================================================================
! Locals
! ==============================================================================

      integer e,f,fdim,i,k,nxz,nface,ifield,testt
      parameter (lfd=lxd*lzd)
      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
! JH111815 legacy rocflu names.
!
! nx,ny,nz : outward facing unit normal components
! fs       : face speed. zero until we have moving grid
! jaco_c   : fdim-D GLL grid Jacobian
! nm       : jaco_c, fine grid
!
! State on the interior (-, "left") side of the face
! rl       : density
! ul,vl,wl : velocity
! tl       : temperature
! al       : sound speed
! pl       : pressure, then phi
! cpl      : rho*cp
! State on the exterior (+, "right") side of the face
! rr       : density
! ur,vr,wr : velocity
! tr       : temperature
! ar       : sound speed
! pr       : pressure
! cpr      : rho*cp

      COMMON /SCRNS/ nx(lfd), ny(lfd), nz(lfd), rl(lfd), ul(lfd),
     >               vl(lfd), wl(lfd), pl(lfd), tl(lfd), al(lfd),
     >               cpl(lfd),rr(lfd), ur(lfd), vr(lfd), wr(lfd),
     >               pr(lfd),tr(lfd), ar(lfd),cpr(lfd), fs(lfd),
     >               jaco_f(lfd),flx(lfd,toteq),jaco_c(lx1*lz1)
      common /dgrad/ jgl(ldg),jgt(ldg)
      real nx, ny, nz, rl, ul, vl, wl, pl, tl, al, cpl, rr, ur, vr, wr,
     >                pr,tr, ar,cpr, fs,jaco_f,flx,jaco_c

!     REAL vf(3)
      real nTol
      character*132 deathmessage
      character*3 cb

      nTol = 1.0E-14

      fdim=ndim-1
      nface = 2*ndim
      nxz   = nx1*nz1
      nxzd  = nxd*nzd
      ifield= 1
      testt=1


!     if (outflsub)then
!        call maxMachnumber
!     endif
      do e=1,nelt
      do f=1,nface

         cb=cbc(f,e,ifield)
         if (cb.ne.'E  '.and.cb.ne.'P  ') then ! cbc bndy

!-----------------------------------------------------------------------
! compute flux for weakly-enforced boundary condition
!-----------------------------------------------------------------------

            !testt = testt+1
            do j=1,nstate
               do i=1,nxz
                  if (abs(qplus(i,f,e,j)) .gt. ntol) then
                  write(6,*) nid,j,i,qplus(i,f,e,j),qminus(i,f,e,j),cb,
     > nstate
                  write(deathmessage,*)  'GS hit a bndy,f,e=',f,e,'$'
! Make sure you are not abusing this error handler
                  call exitti(deathmessage,f)
                  endif
               enddo
            enddo

         else ! cbc(f,e,ifield) == 'E  ' or 'P  ' below; interior face

            !call get_int_ptr (i,nx1,nxd)
! JH111715 now with dealiased surface integrals. I am too lazy to write
!          something better
            !print *,'face number is ',f
            call map_faced(nx,unx(1,1,f,e),nx1,nxd,fdim,0)
            call map_faced(ny,uny(1,1,f,e),nx1,nxd,fdim,0)
            call map_faced(nz,unz(1,1,f,e),nx1,nxd,fdim,0)

            call map_faced(rl,qminus(1,f,e,irho),nx1,nxd,fdim,
     >                        0)
            call map_faced(ul,qminus(1,f,e,iux),nx1,nxd,fdim,0)
            call map_faced(vl,qminus(1,f,e,iuy),nx1,nxd,fdim,0)
            call map_faced(wl,qminus(1,f,e,iuz),nx1,nxd,fdim,0)
            call map_faced(pl,qminus(1,f,e,ipr),nx1,nxd,fdim,0)
            call map_faced(tl,qminus(1,f,e,ithm),nx1,nxd,fdim,
     >                        0)
            call map_faced(al,qminus(1,f,e,isnd),nx1,nxd,fdim,
     >                     0)
            call map_faced(cpl,qminus(1,f,e,icpf),nx1,nxd,fdim,
     >                        0)

            call map_faced(rr,qplus(1,f,e,irho),nx1,nxd,fdim,0)
            call map_faced(ur,qplus(1,f,e,iux),nx1,nxd,fdim,0)
            call map_faced(vr,qplus(1,f,e,iuy),nx1,nxd,fdim,0)
            call map_faced(wr,qplus(1,f,e,iuz),nx1,nxd,fdim,0)
            call map_faced(pr,qplus(1,f,e,ipr),nx1,nxd,fdim,0)
            call map_faced(tr,qplus(1,f,e,ithm),nx1,nxd,fdim,0)
            call map_faced(ar,qplus(1,f,e,isnd),nx1,nxd,fdim,0)
            call map_faced(cpr,qplus(1,f,e,icpf),nx1,nxd,fdim,
     >                        0)

            call invcol3(jaco_c,area(1,1,f,e),wghtc,nxz) !a=b/c
            call map_faced(jaco_f,jaco_c,nx1,nxd,fdim,0)
            call nekcol2(jaco_f,wghtf,nxzd)
            call nekrzero(fs,nxzd) ! moving grid stuff later

            call AUSM_FluxFunction(nxzd,nx,ny,nz,jaco_f,fs,rl,ul,vl,wl,
     >                        pl,al,tl,rr,ur,vr,wr,pr,ar,tr,flx,cpl,cpr)

            call map_faced(pl,qminus(1,f,e,iph),nx1,nxd,fdim,0)
            do j=1,toteq
               call nekcol2(flx(1,j),pl,nxzd)
               call map_faced(flux(1,f,e,j),flx(1,j),nx1,nxd,
     >                        fdim,1)
            enddo

         endif ! cbc(f,e,ifield)
      enddo
      enddo
      !print *,'testt ',testt
      end
