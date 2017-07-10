c----------------------------------------------------------------------------------
      subroutine fluxes_full_field
!-----------------------------------------------------------------------
! JH060314 First, compute face fluxes now that we have the primitive variables
! JH091514 renamed from "surface_fluxes_inviscid" since it handles all fluxes
!          that we compute from variables stored for the whole field (as
!          opposed to one element at a time).
!-----------------------------------------------------------------------
      include 'SIZE'
      include 'DG'
      include 'SOLN'
      include 'CMTDATA'
      include 'INPUT'

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelcmt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*ldim*lfq)
! JH070214 OK getting different answers whether or not the variables are
!          declared locally or in common blocks. switching to a different
!          method of memory management that is more transparent to me.
      common /CMTSURFLX/ fatface(heresize),notyet(hdsize)
      real fatface,notyet
      integer eq
      character*32 cname
      nfq=nx1*nz1*2*ndim*nelt
      nstate = nqq
      ntot1 = nfq*nstate
! where different things live
      iqm =1
      iqp =iqm+nstate*nfq
      iflx=iqp+nstate*nfq

      call fillq(irho,vtrans,fatface(iqm),fatface(iqp))
      call fillq(iux, vx,    fatface(iqm),fatface(iqp)) !- level4.txt
      call fillq(iuy, vy,    fatface(iqm),fatface(iqp))
      call fillq(iuz, vz,    fatface(iqm),fatface(iqp))
      call fillq(ipr, pr,    fatface(iqm),fatface(iqp))
      i_cvars=(iu1-1)*nfq+1
      do eq=1,toteq
         call faceu(eq,fatface(i_cvars)) !- level4.txt
         i_cvars=i_cvars+nfq
      enddo

      call face_state_commo(fatface(iqm),fatface(iqp),nfq,nstate
     >                     ,dg_hndl) !- level4.txt

      call InviscidFlux(fatface(iqm),fatface(iqp),fatface(iflx)
     >                 ,nstate,toteq) !- level4.txt
      return
      end

c----------------------------------------------------------------------------------
      subroutine fillq(ivar,field,qminus,yourface)
      include 'SIZE'
      include 'DG'

      integer ivar! intent(in)
      real field(nx1,ny1,nz1,nelt)! intent(in)
!     real, intent(out)qminus(7,nx1*nz1*2*ldim*nelt) ! gs_op no worky
      real qminus(nx1*nz1*2*ndim*nelt,*)! intent(out)
      real yourface(nx1,nz1,2*ndim,*)
      integer e,f

      nxz  =nx1*nz1
      nface=2*ndim

      call full2face_cmt(nelt,nx1,ny1,nz1,iface_flux,yourface,field) !- level4.f

      do i=1,ndg_face
!        qminus(ivar,i)=yourface(i,1,1,1) ! gs_op_fields no worky yet.
                                          ! tranpose later
         qminus(i,ivar)=yourface(i,1,1,1)
      enddo

      return
      end

c----------------------------------------------------------------------------------

      subroutine compute_rhs_and_dt
!> doxygen comments look like this
      include 'SIZE'
      include 'TOTAL'
      include 'DG'      ! dg_face is stored
      include 'CMTDATA'
      include 'CTIMER'

      integer lfq,heresize,hdsize
      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      parameter (lfq=lx1*lz1*2*ldim*lelcmt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*ldim*lfq)
! not sure yet if viscous surface fluxes can live here yet
      common /CMTSURFLX/ flux(heresize),ViscousStuff(hdsize)
      real ViscousStuff

      COMMON /pnttimers/ pt_time_add, pt_tracking_add
      integer e,eq
      real wkj(lx1+lxd)
      character*32  dumchars
      !parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ jgl(ldg),jgt(ldg)
      real jgl,jgt


      call set_dealias_face

!     filter the conservative variables before start of each
!     time step
! compute primitive vars on the FINE grid. Required to compute conv fluxes.
!        primitive vars = rho, u, v, w, p, T, phi_p
      tbeg1 = dnekclock()
      if(stage.gt.1) call compute_primitive_vars !- level3.txt
      tbeg2 = dnekclock()
      call fluxes_full_field !- level3.txt
      tbeg3 = dnekclock()
      ntot = lx1*ly1*lz1*lelcmt*toteq
      call rzero(res1,ntot)

      nstate=nqq
      nfq=nx1*nz1*2*ndim*nelt
      iqm =1
      iqp =iqm+nstate*nfq
      iflx=iqp+nstate*nfq
      do eq=1,toteq
         ieq=(eq-1)*ndg_face+iflx
         call surface_integral_full(res1(1,1,1,1,eq),flux(ieq)) !- level3.txt
      enddo
      iuj=iflx ! overwritten with [[U]]
      tbeg4 = dnekclock()
      do e=1,nelt
! Get user defined forcing from userf defined in usr file
         ! Bone: call cmtusrf(e)
         ! Bone: if (ifvisc)then
             ! Bone: call compute_gradients(e)
! compute_aux_var will likely not be called if Sij=GdU
             ! Bone: call compute_aux_var(e)
!------------------------------
! NB!!!!! gradu=Sij, NOT dUl/dxk!!!!
!------------------------------
         ! Bone: endif
         do eq=1,toteq
! Now we can start assembling the flux terms in all 5 eqs
! Flux terms are decomposed into h_conv and h_diff
            call assemble_h(e,eq, jgl(i),jgt(i)) !- level3.txt
! compute the volume integral term and assign to res1
            call flux_div_integral(e,eq, jgl(i), jgt(i)) !- level3.txt
            !call surface_integral_elm(e,eq) !- level3.txt
!------------------------------
! JH050615 BR1 ONLY for now
!           if (.not.ifbr1)
!    >      call penalty(flux(iqm),flux(iqp),flux(iuj),e,eq,nstate)
!------------------------------
! Compute the forcing term in each of the 5 eqs
! Add this to the residue
            call compute_forcing(e,eq) !- level3.txt
         enddo
      enddo
      tbeg5 = dnekclock()
      !print *, 'time is ',tbeg2-tbeg1,tbeg3-tbeg2
      !print *,tbeg4-tbeg3,tbeg5-tbeg4
      return
      end

!-----------------------------------------------------------------------
