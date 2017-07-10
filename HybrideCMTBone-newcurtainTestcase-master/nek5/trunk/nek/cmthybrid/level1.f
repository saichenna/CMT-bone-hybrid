      subroutine cmt_nek_advance
c
c     Solve the Euler equations

      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      include 'SOLN'
      include 'GEOM'
      include 'CTIMER'
      include 'CMTDATA'
      include 'CMTTIMERS'
      include 'DG'
      include 'DEALIAS'

      integer e,eq
      character*32 dumchars
      integer lfq,heresize,hdsize
      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      parameter (lfq=lx1*lz1*2*ldim*lelcmt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*ldim*lfq)
      common /dgrad/ d(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real, device :: d_res3(lx1,ly1,lz1,toteq,lelt)
      real, device :: d_u(lx1,ly1,lz1,toteq,lelt)
      real, device :: d_res1(lx1,ly1,lz1,lelt,toteq)
      real, device :: d_bm1(lx1,ly1,lz1,lelt)
      real, device :: d_tcoef(3,3)
      real, device :: d_jgl(ldg)
      real, device :: d_jgt(ldg)
      real, device :: d_flux(hereSize)
      real, device :: d_w(lelt*lwkd)
!      real, device :: d_vx(lx1,ly1,lz1,lelt)
!      real, device :: d_vy(lx1,ly1,lz1,lelt)
!      real, device :: d_vz(lx1,ly1,lz1,lelt)
      real, device :: d_vxd(lxd,lyd,lzd,lelt)
      real, device :: d_vyd(lxd,lyd,lzd,lelt)
      real, device :: d_vzd(lxd,lyd,lzd,lelt)
!      real, device :: d_vtrans(lx1,ly1,lz1,lelt)
!      real, device :: d_pr(lx1,ly1,lz1,lelt)  
      real, device :: d_area(lx1,lz1,6,lelt)  
      real, device :: d_phig(lx1,ly1,lz1,lelt)  
      real, device :: d_iface_flux(lx1*lz1*6,lelt)  
      real, device :: d_totalh(lelt*3*lxd*lyd*lzd)
      real, device :: d_ju1(lelt*lxd*lyd*lzd)     
      real, device :: d_ju2(lelt*lxd*lyd*lzd)
      real, device :: d_ut(lelt*lxd*lyd*lzd)
      real, device :: d_ud(lelt*lxd*lyd*lzd)
      real, device :: d_tu(lelt*lxd*lyd*lzd)     
      real, device :: d_rx(lxd*lyd*lzd,9,lelt)
      real, device :: d_dg(lxd*lyd*lzd)
      real, device :: d_dgt(lxd*lyd*lzd)
      real, device :: d_d(lxd*lyd*lzd)
      real, device :: d_dt(lxd*lyd*lzd)
      real, device :: d_jacmi(lx1*ly1*lz1,lelt)
      real, device :: d_usrf(lx1,ly1,lz1,toteq)
      real, device :: d_vols(lx1,ly1,lz1,lelt,5) 
      real, device :: d_wghtc(lx1*lz1)
      real, device :: d_wghtf(lxd*lzd)
      real, device :: d_unx(lx1,lz1,6,lelt)
      real, device :: d_uny(lx1,lz1,6,lelt)
      real, device :: d_unz(lx1,lz1,6,lelt)
      real, device :: d_cbc(lx1,lz1)
      real, device :: d_all(lelt*6*lxd*lzd,26)
      real, device :: d_jaco_c(lelt*6*lx1*lz1);


      Integer i, ii
      call get_int_ptr(i,lx1,lxd)
      call get_dgl_ptr(ip,lxd,lxd) ! fills dg, dgt
      istate = cudaMemcpy(d_jgl,jgl,ldg,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_jgt,jgt,ldg,cudaMemcpyHosttoDevice)       
      istate = cudaMemcpy(d_dgt,dgt,ldg,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_dg,dg,ldg,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_d,d,ldg,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_dt,dt,ldg,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_u,u,lelt*toteq*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_unx,unx,lelt*6*lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_uny,uny,lelt*6*lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_unz,unz,lelt*6*lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_area,area,lelt*6*lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_wghtc,wghtc,lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_wghtf,wghtf,lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_phig,phig,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_rx,rx,lelt*9*lxd*lyd*lzd,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_jacmi,jacmi,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_usrf,usrf,toteq*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)


      nxyz1=lx1*ly1*lz1
      n = nxyz1*lelcmt*toteq
      nfldpart = ndim*npart

      if(istep.eq.1) call set_tstep_coef !- level2.txt
      if(istep.eq.1) call cmt_flow_ics(ifrestart) !- level2.txt

      istate = cudaMemcpy(d_vols(1,1,1,1,1),vtrans,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_vols(1,1,1,1,2),vx,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_vols(1,1,1,1,3),vy,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_vols(1,1,1,1,4),vz,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_vols(1,1,1,1,5),pr,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_iface_flux,iface_flux,lelt*6*lx1*lz1,         
     > cudaMemcpyHosttoDevice)
      
      istate = cudaMemcpy(d_bm1,bm1,lelt*lx1*ly1*lz1,         
     > cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_tcoef,tcoef,9,cudaMemcpyHosttoDevice)
      istate = cudaMemcpy(d_res1,res1,lelt*lx1*ly1*lz1*toteq,         
     > cudaMemcpyHosttoDevice)


      nstage = 3
      do stage=1,nstage
!         if (stage.eq.1) call nekcopy(res3(1,1,1,1,1),U(1,1,1,1,1),n) !(level2.txt)
         
         if (stage.eq.1) call nekcopywrapper(d_res3,d_u,n) 
         call compute_rhs_and_dt(d_vx,d_vy,d_vz,
        >                   d_vxd,d_vyd,d_vzd,d_u,d_jgl,d_jgt,
        >                   d_w,d_vols,d_flux,d_iface_flux,d_unx, 
        >     d_uny, d_unz, d_area, d_wghtc, d_wghtf, d_cbc,
        >     d_all, d_jaco_c,d_res1,
        >                  d_phig,d_totalh,d_ur,d_us,
        > d_ut,d_ud,d_tu,d_rx,d_dg,d_dgt,
        > d_jacmi,d_bm1,d_usrf,d_d,d_dt)
         call calculateuwrapper(d_u,d_bm1,d_tcoef,d_res3,d_res1,stage,
     >                     nelt,nxyz1,toteq)
         !print *,"u is ",u(1,1,1,1,1)
C         do e=1,nelt
C            do eq=1,toteq
C            do i=1,nxyz1
c multiply u with bm1 as res has been multiplied by bm1 in compute_rhs
C               u(i,1,1,eq,e) = bm1(i,1,1,e)*tcoef(1,stage)
C     >                     *res3(i,1,1,eq,e)+bm1(i,1,1,e)*
C     >                     tcoef(2,stage)*u(i,1,1,eq,e)-
C     >                     tcoef(3,stage)*res1(i,1,1,e,eq)
c-----------------------------------------------------------------------
c this completely stops working if B become nondiagonal for any reason.
C               u(i,1,1,eq,e) = u(i,1,1,eq,e)/bm1(i,1,1,e)
c that completely stops working if B become nondiagonal for any reason.
!-----------------------------------------------------------------------
C            enddo
C            enddo
C         enddo
      enddo
      u = d_u
 101  format(4(2x,e18.9))
      return
      end

      subroutine nek_cmt_init
      include 'SIZE'
      include 'TOTAL'
      include 'DG'
      if (nio.eq.0) write(6,*)'Set up CMT-Nek'
      if (toteq.ne.5) then
         if (nio.eq.0) write(6,*)'toteq is low ! toteq = ',toteq
         if (nio.eq.0) write(6,*) 'Reset toteq in SIZE to 5'
         call exitt
      endif
      if (lelcmt.ne.lelt) then
         if (nio.eq.0) write(6,*)'ERROR! lelcmt is not same as lelt '
         if (nio.eq.0) write(6,*) 'lelcmt=',lelcmt,' lelt=',lelt
         call exitt
      endif
      call setup_cmt_commo

c     call setup_cmt_param
      return
      end

