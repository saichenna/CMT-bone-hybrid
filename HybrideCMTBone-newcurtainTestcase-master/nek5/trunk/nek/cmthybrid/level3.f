      subroutine get_shared_faces               !CALLED FROM NEK_CMT_INIT

      include 'SIZE'
      include 'INPUT' ! for BC
      include 'PARALLEL' ! for GLLNID

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
c     common /shareddata/ isGPU, num_sh, shArray(2, lelt*6)
      integer isGPU, num_sh, num_cores, shArray(2, lelt*6)
      common /shareddata/ isGPU, num_sh, num_cores, shArray

      integer e, f, ieg, shel, shf
      character c*1

      num_sh = 0
      !print *, "faces shared with element"
      do e=1, nelt
          do f = 1, 2*ndim
             c = CBC(f, e, 1)
             if ((c .eq. 'P') .OR. (c .eq. 'E')) then  !consider only periodic or regular faces
                 shel = BC(1, f, e, 1)                 !shared element
                 shf = BC(2, f, e, 1)                  !shared face
                 if (NID .NE. GLLNID(shel)) then
                     !add to array
                     !print *, NID, LGLEL(e), f, shel, shf, GLLNID(shel)
                     num_sh = num_sh + 1
                     shArray(1, num_sh) = shel
                     shArray(2, num_sh) = shf
                 endif
             endif
          enddo
      enddo
      !print *, "done"

      end subroutine 
      subroutine nekrzero(a,n)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      return
      END

!-----------------------------------------------------------------------

      subroutine set_dealias_face

!-----------------------------------------------------------------------
! JH111815 needed for face Jacobian and surface integrals
!-----------------------------------------------------------------------

      include 'SIZE'
      include 'INPUT' ! for if3d
      include 'GEOM'  ! for ifgeom
      include 'TSTEP' ! for istep
      include 'WZ'    ! for wxm1
      include 'DG'    ! for facewz

      integer ilstep
      save    ilstep
      data    ilstep /-1/

      if (.not.ifgeom.and.ilstep.gt.1) return  ! already computed
      if (ifgeom.and.ilstep.eq.istep)  return  ! already computed
      ilstep = istep

      call zwgl(zptf,wgtf,nxd) ! comes from ../speclib.f - no need to 
                               ! migrate ! to GPU.

      if (if3d) then
         k=0
         do j=1,ny1
         do i=1,nx1
            k=k+1
            wghtc(k)=wxm1(i)*wzm1(j)
         enddo
         enddo
         k=0
         do j=1,nyd
         do i=1,nxd
            k=k+1
            wghtf(k)=wgtf(i)*wgtf(j)
         enddo
         enddo
      else
         call nekcopy(wghtc,wxm1,nx1)
         call nekcopy(wghtf,wgtf,nxd)
      endif

      return
      end
c----------------------------------------------------------------------------------
      subroutine compute_primitive_vars
      include 'SIZE'
      include 'INPUT'
      include 'CMTDATA'
      include 'SOLN'
      include 'DEALIAS'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      parameter (lxyz=lx1*ly1*lz1)
      common /ctmp1/ energy(lx1,ly1,lz1),scr(lx1,ly1,lz1)
      integer e, eq

      nxyz= nx1*ny1*nz1
      ntot=nxyz*nelt
      !print *,irpu,irpv,irpw,irg

      do e=1,nelt
         call nekinvcol3(vx(1,1,1,e),u(1,1,1,irpu,e),
     >                   u(1,1,1,irg,e),nxyz) !- level4.txt
         call nekinvcol3(vy(1,1,1,e),u(1,1,1,irpv,e),
     >                   u(1,1,1,irg,e),nxyz)
         if (if3d)
     >     call nekinvcol3(vz(1,1,1,e),u(1,1,1,irpw,e),
     >                   u(1,1,1,irg,e),nxyz)
      enddo

c     gas_right_boundary = exp(TIME/2.0)
c     do e=1,nelt
c        x_left_boundary = xerange(1,1,e)
c        if (x_left_boundary .lt. gas_right_boundary) then
c            call rone(vx(1,1,1,e), nxyz)
c        else
c            call rzero(vx(1,1,1,e), nxyz)
c        endif
         
         call rzero(vx(1,1,1,e), nxyz)
         call rone(vy(1,1,1,e), nxyz)
         call rzero(vz(1,1,1,e), nxyz)
c     enddo


      call set_convect_cons (vxd,vyd,vzd,vx,vy,vz)
      return
      end

c----------------------------------------------------------------------------------
      subroutine surface_integral_full(vol,flux)
! Integrate surface fluxes for an entire field. Add contribution of flux
! to volume according to add_face2full_cmt
      include 'SIZE'
      include 'GEOM'
      include 'DG'
      include 'CMTDATA'
      real vol(nx1*ny1*nz1*nelt),flux(*)
      integer e,f

      nxz=nx1*nz1
      nface=2*ldim
      k=0

! JH030915 As qminus and qplus grow and get more flexible, throttling by phig
!          should be done via qminus and NOT the way it is done here. We don't
!          need icpvars anymore, and, most importantly ViscousFlux ALREADY HAS PHI!!!!!!
      l = 0
      do e=1,nelt
      do f=1,nface
         call nekfacind(i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)
         k = 0
         do iz=k0,k1
         do iy=j0,j1
         do ix=i0,i1
            l = l + 1
            k = k + 1
            flux(l)=flux(l)*area(k,1,f,e)*phig(ix,iy,iz,e)
         enddo
         enddo
         enddo
      enddo
      enddo
      call add_face2full_cmt(nelt,nx1,ny1,nz1,iface_flux,vol,flux) !- level4.txt

      return
      end


c----------------------------------------------------------------------------------
      subroutine assemble_h(e,eq_num,ba,ab)
      include 'SIZE'
      include 'CMTDATA'
      include 'INPUT'

      integer  e,eq_num

!     !This subroutine will compute the convective and diffusive
!     !components of h
      call nekrzero(totalh,3*lxd*lyd*lzd)
      call evaluate_conv_h(e,eq_num,ba,ab) !- level4.txt
      ! Bone: if (ifvisc.and.eq_num .gt. 1) call evaluate_diff_h(e,eq_num)
      call add_conv_diff_h

      return
      end


c----------------------------------------------------------------------------------
      subroutine flux_div_integral(e,eq)
      include  'SIZE'
      include  'INPUT'
      include  'GEOM'
      include  'MASS'
      include  'CMTDATA'

      integer  e,eq
      integer  dir
      parameter (ldd=lxd*lyd*lzd)
      parameter (ldg=lxd**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      nrstd=ldd
      nxyz=nx1*ny1*nz1
      call get_dgl_ptr(ip,nxd,nxd) ! fills dg, dgt
      mdm1=nxd-1

      call nekrzero(ur,nrstd)
      call nekrzero(us,nrstd)
      call nekrzero(ut,nrstd)
      call nekrzero(ud,nrstd)
      call nekrzero(tu,nrstd)

      j0=0
      do j=1,ndim
         j0=j0+1
         call nekadd2col2(ur,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      do j=1,ndim
         j0=j0+1
         call nekadd2col2(us,totalh(1,j),rx(1,j0,e),nrstd)
      enddo
      if (if3d) then
         do j=1,ndim
            j0=j0+1
            call nekadd2col2(ut,totalh(1,j),rx(1,j0,e),nrstd)
         enddo
         call local_grad3_t(ud,ur,us,ut,mdm1,1,dg(ip),dgt(ip),wkd)
      else
         call local_grad2_t(ud,ur,us,   mdm1,1,dg(ip),dgt(ip),wkd)
      endif

      call intp_rstd(tu,ud,nx1,nxd,if3d,1)
      call neksub2(res1(1,1,1,e,eq),tu,nxyz)

      return
      return
      end

c----------------------------------------------------------------------------------
      subroutine compute_forcing(e,eq_num)
      include  'SIZE'
      include  'INPUT'
      include  'GEOM'
      include  'MASS'
      include  'SOLN'
      include  'CMTDATA'
      include  'DEALIAS'

      integer e,eq_num
      parameter (ldd=lxd*lyd*lzd)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju
      nrstd=ldd
      nxyz=nx1*ny1*nz1
      call nekrzero(ud,nxyz)
      if(eq_num.ne.1.and.eq_num.ne.5)then
        if(eq_num.eq.2)then
           j=1
        elseif(eq_num.eq.3)then
           j=2
        elseif(eq_num.eq.4)then
           j=2
           if(ldim.eq.3) j=3
        endif
c       write(6,*)'enter  compute_forcing ', j
        call nekgradl_rst(ur,us,ut,phig(1,1,1,e),lx1,if3d) ! navier1 - level4.txt
        if (if3d) then
           j0=j+0
           j3=j+3
           j6=j+6
           do i=1,nrstd   ! rx has mass matrix and Jacobian on fine mesh
              ud(i)=rx(i,j0,e)*ur(i)+rx(i,j3,e)*us(i)+rx(i,j6,e)*ut(i)
           enddo
        else
           j0=j+0
           j2=j+2
           do i=1,nrstd   ! rx has mass matrix and Jacobian on fine mesh
              ud(i)=rx(i,j0,e)*ur(i)+rx(i,j2,e)*us(i)
           enddo
        endif
        if (eq_num.eq.4.and.ldim.eq.2)then

        else
           call nekcol2(ud,pr(1,1,1,e),nxyz)
           call nekcopy(convh(1,1),ud,nxyz)
           call nekcol2(convh(1,1),jacmi(1,e),nxyz)
           call nekcol2(convh(1,1),bm1(1,1,1,e),nxyz)  ! res = B*res
           call neksub2(res1(1,1,1,e,eq_num),convh(1,1),nxyz) !- level4.txt
           call neksubcol3(res1(1,1,1,e,eq_num),usrf(1,1,1,eq_num)
     $                  ,bm1(1,1,1,e),nxyz) !- level4.txt
        endif
      elseif(eq_num.eq.5)then
           call neksubcol3(res1(1,1,1,e,eq_num),usrf(1,1,1,eq_num)
     $                  ,bm1(1,1,1,e),nxyz) !- level4.txt
      endif
      return
      end

!-----------------------------------------------------------------------

      subroutine setup_cmt_gs(dg_hndl,nx,ny,nz,nel,melg,vertex,gnv,gnf)

!     Global-to-local mapping for gs

      include 'SIZE'
      include 'TOTAL'

      integer   dg_hndl
      integer   vertex(*)

      integer*8 gnv(*),gnf(*),ngv
      integer*8 nf

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      mx = nx+2
      call set_vert(gnv,ngv,mx,nel,vertex,.false.) ! lives in navier8.f

      mz0 = 1
      mz1 = 1
      if (if3d) mz0 = 0
      if (if3d) mz1 = nz+1
      call iface_vert_int8cmt(nx,ny,nz,gnf,gnv,mz0,mz1,nelt)

      nf = nx*nz*2*ndim*nelt !total number of points on faces
      call gs_setup(dg_hndl,gnf,nf,nekcomm,np)

      return
      end

c-------------------------------------------------------------------------------

      subroutine cmt_set_fc_ptr(nel,nx,ny,nz,nface,iface)

!     Set up pointer to restrict u to faces ! NOTE: compact
! JH062314 Now 2D so we can strip faces by element and not necessarily
!          from the whole field

      include 'SIZE'
      include 'TOTAL'

      integer nx, ny, nz, nel
      integer nface,iface(nx*nz*2*ldim,*)
      integer e,f,ef

      call dsset(nx,ny,nz) ! set skpdat. lives in connect1.f

      nxyz = nx*ny*nz
      nxz  = nx*nz
      nfpe = 2*ndim
      nxzf = nx*nz*nfpe ! red'd mod to area, unx, etc.

      do e=1,nel
      do f=1,nfpe

         ef     = eface(f)
         js1    = skpdat(1,f)
         jf1    = skpdat(2,f)
         jskip1 = skpdat(3,f)
         js2    = skpdat(4,f)
         jf2    = skpdat(5,f)
         jskip2 = skpdat(6,f)

         i = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1

            i = i+1
            k = i+nxz*(ef-1)           ! face numbering
            iface(k,e) = j1+nx*(j2-1)  ! cell numbering

         enddo
         enddo

      enddo
      enddo
      nface = nxzf*nel

      return
      end

!-----------------------------------------------------------------------

