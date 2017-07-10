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

      integer e,eq
      character*32 dumchars

      nxyz1=lx1*ly1*lz1
      n = nxyz1*lelcmt*toteq
      nfldpart = ndim*npart

      if(istep.eq.1) call set_tstep_coef !- level2.txt
      if(istep.eq.1) call cmt_flow_ics(ifrestart) !- level2.txt
      if(istep.eq.1) call usr_particles_init

      nstage = 3
      do stage=1,nstage
         if (stage.eq.1) call nekcopy(res3(1,1,1,1,1),U(1,1,1,1,1),n) !(level2.txt)

         call compute_rhs_and_dt !- level2.txt
         call usr_particles_solver
         tbeg = dnekclock()
         do e=1,nelt
            do eq=1,toteq
            do i=1,nxyz1
c multiply u with bm1 as res has been multiplied by bm1 in compute_rhs
               u(i,1,1,eq,e) = bm1(i,1,1,e)*tcoef(1,stage)
     >                     *res3(i,1,1,eq,e)+bm1(i,1,1,e)*
     >                     tcoef(2,stage)*u(i,1,1,eq,e)-
     >                     tcoef(3,stage)*res1(i,1,1,e,eq)
c              u(i,1,1,eq,e) = bm1(i,1,1,e)*u(i,1,1,eq,e) - DT *
c    >                        (c1*res1(i,1,1,e,eq) + c2*res2(i,1,1,e,eq)
c    >                       + c3*res3(i,1,1,e,eq))
c-----------------------------------------------------------------------
c this completely stops working if B become nondiagonal for any reason.
               u(i,1,1,eq,e) = u(i,1,1,eq,e)/bm1(i,1,1,e)
c that completely stops working if B become nondiagonal for any reason.
!-----------------------------------------------------------------------
            enddo
            enddo
         enddo
         tend = dnekclock()
         !print *,'calc u time',tend-tbeg
      enddo
 101  format(4(2x,e18.9))
      return
      end

c-----------------------------------------------------------------------
      subroutine nek_solve_cpu2
      end subroutine
c-----------------------------------------------------------------------
      subroutine nek_solve_cpu

      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'CTIMER'

      real*4 papi_mflops
      integer*8 papi_flops
      integer modstep
      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      integer gfirst, inoassignd, resetFindpts, pload
      integer reinit_step  !added by keke
      integer counter !added by keke
      integer last_kstep !added by keke
      real diff_time

      call nekgsync()
      reinit_step=0
      diff_time = 0.0
      counter = 0
      last_kstep = 0

      if (instep.eq.0) then
        if(nid.eq.0) write(6,'(/,A,/,A,/)') 
     &     ' nsteps=0 -> skip time loop',
     &     ' running solver in post processing mode'
      else
        if(nio.eq.0) write(6,'(/,A,/)') 'Starting time loop cpu ...'
      endif

      isyc  = 0
      itime = 0
      if(ifsync) isyc=1
      itime = 1
      call nek_comm_settings(isyc,itime)

      call nek_comm_startstat()

      istep  = 0
      msteps = 1

      !print *, "CPU nsteps", nsteps, isyc

      do kstep=1,nsteps,msteps
         call nek__multi_advance(kstep,msteps)
         !call process_cpu_particles
         call prepost (.false.,'his')
         call in_situ_check()
         resetFindpts = 0 
         if (lastep .eq. 1) goto 1001

c        modstep = mod(kstep, 500)
c        if (modstep .eq. 0) then
c          resetFindpts = 1
c          call reinitialize
c          call printVerify
c        endif

c        auto load balancing
         if(nid .eq. 0) then
         if(kstep .le. reinit_step+10) then !for the first 10 step after
                                            !rebalance, pick the minimum
                                            !one as the init_time
            if((INIT_TIME .gt. TTIME_STP) .and. (TTIME_STP .ne. 0)) then
                INIT_TIME = TTIME_STP
            endif
         else if(kstep .gt. reinit_step+100) then
            diff_time = (TTIME_STP-INIT_TIME)/INIT_TIME
               if(nid .eq. 0) then
               print *, "nid:", nid, "ttime_stp:", TTIME_STP, INIT_TIME
     $           ,diff_time
               endif
         endif
         endif

         call bcast(diff_time, 8)
         if (diff_time .gt. 0.1) then
c           print *, "diff_time:", diff_time, "counter:", counter, nid
c    >         , last_kstep
            if (last_kstep .eq. 0) then
                counter = counter + 1
            else if((counter .le. 2) .and.
     $                     (last_kstep .eq. kstep-1))then
                counter = counter + 1
            else
                counter = 0
            endif
            last_kstep = kstep
            if (counter .gt. 2) then
                !print *, "into the reinit, nid:", nid, "diff_time:",
     $            !diff_time
                resetFindpts = 1
                call reinitialize
                call printVerify
                reinit_step = kstep
                if(nid .eq. 0) then
                   print *, "reinitilize, reiniti_step:", reinit_step
                endif
                diff_time = 0.0
                INIT_TIME = 100
                counter = 0
            endif
         endif

         !print *, "Finished CPU"
      enddo
 1001 lastep=1


      call nek_comm_settings(isyc,0)

      call comment

c     check for post-processing mode
      if (instep.eq.0) then
         nsteps=0
         istep=0
         if(nio.eq.0) write(6,*) 'call userchk'
         call userchk
         if(nio.eq.0) write(6,*) 'done :: userchk'
         call prepost (.true.,'his')
      else
         if (nio.eq.0) write(6,'(/,A,/)') 
     $      'end of time-step loop cpu' 
      endif

      RETURN
      END
c-----------------------------------------------------------------------
      subroutine nek_advance

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common /cgeom/ igeom

      call nekgsync
      if (iftran) call settime
      if (ifmhd ) call cfl_check
      call setsolv
      call comment

      if (ifcmt) then
         if (nio.eq.0.and.istep.le.1) write(6,*) 'CMT branch active CPU'
         call cmt_nek_advance
         return
      endif

      if (ifsplit) then   ! PN/PN formulation

         igeom = 1
         if (ifheat)          call heat     (igeom)
         call setprop
         call qthermal
         igeom = 1
         if (ifflow)          call fluid    (igeom)
         if (param(103).gt.0) call q_filter(param(103))
         call setup_convect (2) ! Save convective velocity _after_ filter

      else                ! PN-2/PN-2 formulation

         call setprop
         do igeom=1,ngeom

            if (igeom.gt.2) call userchk_set_xfer

            if (ifgeom) then
               call gengeom (igeom)
               call geneig  (igeom)
            endif

            if (ifneknekm.and.igeom.eq.2) call multimesh_create

            if (ifmhd) then
               if (ifheat)      call heat     (igeom)
                                call induct   (igeom)
            elseif (ifpert) then
               if (ifbase.and.ifheat)  call heat          (igeom)
               if (ifbase.and.ifflow)  call fluid         (igeom)
               if (ifflow)             call fluidp        (igeom)
               if (ifheat)             call heatp         (igeom)
            else  ! std. nek case
               if (ifheat)             call heat          (igeom)
               if (ifflow)             call fluid         (igeom)
               if (ifmvbd)             call meshv         (igeom)
            endif

            if (igeom.eq.ngeom.and.param(103).gt.0) 
     $          call q_filter(param(103))

            call setup_convect (igeom) ! Save convective velocity _after_ filter

         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine nek__multi_advance(kstep,msteps)
      include 'SIZE'
      include 'TSTEP'

      !include 'f77papi.h'
      !include 'TOTAL'

      !integer NUM_EVENTS
      !parameter (NUM_EVENTS=2)
      !integer*8 values(NUM_EVENTS)
      !integer ierr
      !integer Events(NUM_EVENTS)

      !Events(1) = PAPI_L2_DCM
      !Events(2) = PAPI_FP_OPS
      !values(1) = 0
      !values(2) = 0

      !call PAPIF_start_counters(Events, NUM_EVENTS, ierr)
      !if (ierr /= PAPI_OK) then
        !print*, ierr
        !print*, "Could not start counter\n";
      !endif

      do i=1,msteps
         istep = istep+i
         call nek_advance

         if (ifneknek) call userchk_set_xfer
         if (ifneknek) call bcopy
         if (ifneknek) call chk_outflow

      enddo
      !call PAPIF_read_counters(values, NUM_EVENTS, ierr)
      !if (ierr /= PAPI_OK) then
        !print*, ierr
        !print*, "Could not read counter\n";
      !endif
!
      !print *, "Floating point operations=", values(2)


      return
      end
c-----------------------------------------------------------------------
!----------------------------------------------------------------------

      subroutine userchk_cmtbone
      include 'SIZE'

      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni

!     pt_timers - particle timer
!     scrt_timers - scratch timer (IO)
      real    pt_timers(10), scrt_timers(10)
      common /trackingtime/ pt_timers, scrt_timers

         nr = lr     ! Mandatory for proper striding
         ni = li     ! Mandatory

         call rzero(rpart,lr*lpart)
         call izero(ipart,li*lpart)
         !call baryweights
         !call init_stokes_particles   (rpart,nr,ipart,ni,n) ! n initialized here
         call init_interpolation
         call place_particles  ! n initialized here

      end

c---------------------------------------------------------------------------
      subroutine init_interpolation_cmtbone
      include 'SIZE'
      include 'INPUT'
c
c     calculates the barycentric lagrange weights
c
      common /BARYPARAMS/ xgll, ygll, zgll, wxgll, wygll, wzgll
      real xgll(lx1), ygll(ly1), zgll(lz1),
     >     wxgll(lx1), wygll(ly1), wzgll(lz1)


      pi  = 4.0*atan(1.)

c     get gll points in all directions
      call zwgll(xgll,wxgll,lx1)
      call zwgll(ygll,wygll,ly1)
      call rone(zgll,lz1)
      if(if3d) call zwgll(zgll,wzgll,lz1)
c     set all weights to ones first
      call rone(wxgll,lx1)
      call rone(wygll,ly1)
      call rone(wzgll,lz1)
c     calc x bary weights
      do j=1,lx1
         do k=1,lx1
            if (j .NE. k) then
               wxgll(j) = wxgll(j)/(xgll(j) - xgll(k))
            endif
         enddo
      enddo
c     calc y bary weights
      do j=1,ly1
         do k=1,ly1
            if (j .NE. k) then
               wygll(j) = wygll(j)/(ygll(j) - ygll(k))
            endif
         enddo
      enddo
c     calc z bary weights
      do j=1,lz1
         do k=1,lz1
            if (j .NE. k) then
               wzgll(j) = wzgll(j)/(zgll(j) - zgll(k))
            endif
         enddo
      enddo
      return
      end

c---------------------------------------------------------------------------

      subroutine place_particles_cmtbone
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'

      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni
      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid

      real   xerange(2,3,lelt)
      common /elementrange/ xerange
      real   xdrange(2,3)
      common /domainrange/ xdrange

      integer nw
      common /particlenumber/ nw

      integer gfirst, inoassignd, resetFindpts, pload
      integer nw,nwe, remainder, i, ip, e, nl, k, npass, m
      real    rho_p, dp, tau_p, mu_0
      real cenPos(3)     !x0, y0, z0   ! central point for particles
      real tempa, tempb, tempc, tempd
      common /myparth/ i_fp_hndl, i_cr_hndl
      logical partl

      nl = 0
      pi  = 4.0*atan(1.)

      call set_part_pointers

      rho_p     = 1130       ! kg/m^3, example particle density (steel)
      mu_0      = 18.27E-6   ! Pa s, inital fluid viscosity
      dp        = 1.92e-3
c     tau_p     = dp**2*rho_p/18.0d+0/mu_0
      tau_p     = 1e-16      ! tracer if 1e-16

      nw = param(72)   ! number of particles

      if(nid .eq. 0) then
        print *, "nw: ", nw
c       random assign x0, y0, z0 within the domain
        cenPos(1) = rand(70886)*(xdrange(2,1)-xdrange(1,1))
        cenPos(2) = rand(0)*(xdrange(2,2)-xdrange(1,2))
        cenPos(3) = rand(0)*(xdrange(2,3)-xdrange(1,3))
c       print *, 'xdrange: ', xdrange(1,1), xdrange(2,1), 
c    >     xdrange(1,2), xdrange(2,2), xdrange(1,3), xdrange(2,3)
c       print *, 'init cenPos :', cenPos
       endif

c      broadcast cenPos to other processor
       call bcast(cenPos, 8*3)
c      print *, 'nid: ', nid, 'after broadcast, cenPos:', cenPos

c      calculate # particles each processor should initialize
       nwe = nw/np
       remainder = mod(nw,np)
       if(nid .lt. remainder) nwe = nwe + 1

       print *, 'nid: ', nid, 'nwe: ', nwe
       call srand(7086+nid*1000)
        do i = 1, nwe
c          randomly assign the line direction, and normalize it to 1
c          curtain test case
c           rpart(jx, i) = rand(0)
c           rpart(jy, i) = rand(0)
c           rpart(jz, i) = rand(0)

c          new curtain test case
           rpart(jx, i) = rand(0)*10.0
           rpart(jy, i) = rand(0)*0.1
           rpart(jz, i) = rand(0)
           ! Dont delete the next 20 lines as it may be needed for other tests
           ! Eventually move place particles to the examples directory
           !tempa = (rand(0)-0.5)*2  !generate from (-1,1)
           !tempb = (rand(0)-0.5)*2  !generate from (-1,1)
           !tempc = (rand(0)-0.5)*2  !generate from (-1,1)
           !tempd = sqrt(tempa**2+tempb**2+tempc**2)

           !rpart(jaa, i) = tempa/tempd
           !rpart(jab, i) = tempb/tempd
           !rpart(jac, i) = tempc/tempd !now a^2+b^2+c^2=1
           !rpart(jad, i) = rand(0)*1.0/8*(xdrange(2,1)-xdrange(1,1))

           !rpart(jx, i) = cenPos(1) + rpart(jaa, i)*rpart(jad, i)
           !rpart(jy, i) = cenPos(2) + rpart(jab, i)*rpart(jad, i)
           !rpart(jz, i) = cenPos(3) + rpart(jac, i)*rpart(jad, i)
           rpart(jar,i) = tau_p          ! Tracer? or is it 0?
c          print *, 'particle i:', i, nid, rpart(jaa, i), rpart(jab, i),
c    >             rpart(jac, i), rpart(jad, i)
        enddo
       n = nwe



c     check if zstart and zlen is alright for a 2d case
      if (.not. if3d) then
          if (abs(zstart-1.0) .gt. 1E-16) then
             write(6,*)'***particle zstart is not right for 2d case'
             call exitt
          elseif(abs(zlen) .gt. 1E-16) then
             write(6,*)'***particle zlen is not right for 2d case'
             call exitt
         endif
      endif
      call update_particle_location(1)
      call move_particles_inproc
c     print *, 'finish move_particles_inproc'
      !call particles_solver_nearest_neighbor
c     print *, 'finish particles_solver_nearest_neighbor'
      resetFindpts = 1
      call reinitialize
      call printVerify

c     set local particle id
      do i = 1, n
         ipart(jpid2, i) = i
      enddo

      return
      end
c-----------------------------------------------------------------------

c----------------------------------------------------------------------
c     effeciently move particles between processors routines
c----------------------------------------------------------------------
      subroutine move_particles_inproc_cmtbone
c     Interpolate fluid velocity at current xyz points and move
c     data to the processor that owns the points.
c     Input:    n = number of points on this processor
c     Output:   n = number of points on this processor after the move
c     Code checks for n > lpart and will not move data if there
c     is insufficient room.
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /myparth/ i_fp_hndl, i_cr_hndl
      integer isGPU, num_sh, num_cores, shArray(2, lelt*6)
      common /shareddata/ isGPU, num_sh, num_cores, shArray

      parameter (lrf=4+ldim,lif=5+5)
      real               rfpts(lrf,lpart)
      common /fptspartr/ rfpts
      integer            ifpts(lif,lpart),fptsmap(lpart)
      common /fptsparti/ ifpts,fptsmap

      real   xerange(2,3,lelt)
      common /elementrange/ xerange
      real   xdrange(2,3)
      common /domainrange/ xdrange

      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid

      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      integer gfirst, inoassignd, resetFindpts, pload
      integer nw
      common /particlenumber/ nw


      integer icalld1
      save    icalld1
      data    icalld1 /0/

      logical partl         ! This is a dummy placeholder, used in cr()
      nl = 0                ! No logicals exchanged


c     if ((icalld1.eq.0) .or. (resetFindpts .eq. 1)) then
c        tolin = 1.e-12
c        if (wdsize.eq.4) tolin = 1.e-6
c        call intpts_setup  (tolin,i_fp_hndl)
c        call crystal_setup (i_cr_hndl,nekcomm,np)
c        icalld1 = icalld1 + 1
c        resetFindpts = 0
c        added by keke
c        print *, 'before transfer nid: ', nid, '# particles: ', n
c        end added by keke
c     endif

      call interp_comm_part() !in box.usr.orignal, moved to here

      call particles_in_nid(fptsmap,rfpts,lrf,ifpts,lif,nfpts)

      call findpts(i_fp_hndl !  stride     !   call findpts( ihndl,
     $           , ifpts(jrc,1),lif        !   $             rcode,1,
     $           , ifpts(jpt,1),lif        !   &             proc,1,
     $           , ifpts(je0,1),lif        !   &             elid,1,
     $           , rfpts(jr ,1),lrf        !   &             rst,ndim,
     $           , rfpts(jd ,1),lrf        !   &             dist,1,
     $           , rfpts(jx ,1),lrf        !   &             pts(    1),1,
     $           , rfpts(jy ,1),lrf        !   &             pts(  n+1),1,
     $           , rfpts(jz ,1),lrf ,nfpts)    !   &             pts(2*n+1),1,n)

      nmax = iglmax(n,1)
      if (nmax.gt.lpart) then
         if (nid.eq.0) write(6,1) nmax,lpart
    1    format('WARNING: Max number of particles:'
     $   i9,'.  Not moving because lpart =',i9,'.')
      else
c        copy rfpts and ifpts back into their repsected positions in rpart and ipart
         call update_findpts_info(rfpts,lrf
     $                       ,ifpts,lif,fptsmap,nfpts)
c        Move particle info to the processor that owns each particle
c        using crystal router in log P time:

c        print *, 'nid: ', nid, 'nfpts: ', nfpts
         jps = jpid1-1     ! Pointer to temporary proc id for swapping
         do i=1,n        ! Can't use jpt because it messes up particle info
            ipart(jps,i) = ipart(jpt,i)
         enddo
         
         call crystal_tuple_transfer(i_cr_hndl,n,lpart
     $              , ipart,ni,partl,nl,rpart,nr,jps)
         
c        Sort by element number - for improved local-eval performance
         call crystal_tuple_sort    (i_cr_hndl,n
     $              , ipart,ni,partl,nl,rpart,nr,je0,1)
      endif


c     if (isGPU) then
c         call transferParticlesToGPU(rpart, ipart)
          !istate = cudaMemcpy(d_rpart, rpart, HostToDevice)
          !istate = cudaMemcpy(d_ipart, ipart, HostToDevice)
c     endif
c     Interpolate (locally, if data is resident).
      call interp_props_part_location !need to fix this, don't know why crashes here

c        added by keke
c     if(mod(nid, 5) .eq. 0) then
         print *, 'nid:', nid, '#particles:', n, 'nelt'
     $        ,nelt, 'nw', nw, 'nelgt', nelgt
     $        ,n+nelt*ceiling(nw*1.0)/nelgt
c     endif
c        end added by keke
      return
      end
c-----------------------------------------------------------------------
      subroutine particles_in_nid_cmtbone
     $  (fptsmap,rfpts,nrf,ifpts,nif,nfpts)
      include 'SIZE'
      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni

      real    rfpts(nrf,*)
      integer ifpts(nif,*),fptsmap(*)

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid

      nfpts = 0
      do ip = 1,n
         xloc = rpart(jx,ip)
         yloc = rpart(jy,ip)
         zloc = rpart(jz,ip)
         itest = 0
         do ie=1,nelt
            if (xloc.ge.xerange(1,1,ie).and.xloc.le.xerange(2,1,ie))then
            if (yloc.ge.xerange(1,2,ie).and.yloc.le.xerange(2,2,ie))then
            if (zloc.ge.xerange(1,3,ie).and.zloc.le.xerange(2,3,ie))then
                ipart(je0 ,ip) = ie-1
                ipart(jrc ,ip) = 0
                ipart(jpt ,ip) = nid
                rpart(jd  ,ip) = 1.0
                rloc = -1.0 + 2.0*(xloc - xerange(1,1,ie))/
     $                 (xerange(2,1,ie)-xerange(1,1,ie))
                sloc = -1.0 + 2.0*(yloc - xerange(1,2,ie))/
     $                 (xerange(2,2,ie)-xerange(1,2,ie))
                tloc = -1.0 + 2.0*(zloc - xerange(1,3,ie))/
     $                 (xerange(2,3,ie)-xerange(1,3,ie))
                rpart(jr  ,ip) = rloc
                rpart(jr+1,ip) = sloc
                rpart(jr+2,ip) = tloc
                itest = 1
                goto 123
            endif
            endif
            endif
         enddo
         if (itest.eq.0)then
            nfpts = nfpts + 1
            fptsmap(nfpts) = ip
            call copy (rfpts(1,nfpts),rpart(1,ip),nrf)
            call icopy(ifpts(1,nfpts),ipart(1,ip),nif)
            if(nfpts.gt.lpart)then
               write(6,*)'Too many points crossing over ',
     $                      nfpts,lpart,nid
               call exitt
            endif
         endif
123      continue
      enddo
      return
      end
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
      subroutine particles_solver_nearest_neighbor_cmtbone
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'

      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni


      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid
      integer pdimc(3)

      real   xerange(2,3,lelt)
      common /elementrange/ xerange
      real   xdrange(2,3)
      common /domainrange/ xdrange

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /myparth/ i_fp_hndl, i_cr_hndl

      real               rfpts(lr,lpart),rtpts(lr,lpart)
      integer            ifpts(li,lpart),fptsmap(lpart),itpts(lr,lpart)

      logical partl         ! This is a dummy placeholder, used in cr()
      real d2chk,pdist,pxyzg(3,7)
      integer iitest(3), inpt(3,lpart), itest


      nl = 0                ! No logicals exchanged
      d2chk = 0.2
      d3    = 0.2

      nfpts = 0
      ntpts = 0
      do ip = 1,n
         ipart(jai,ip) = ipart(jpnn,ip)


         iitest(1) = 0
         iitest(2) = 0
         iitest(3) = 0
         xloc = rpart(jx,ip)
         yloc = rpart(jy,ip)
         zloc = rpart(jz,ip)
         iie = ipart(je0,ip) + 1

         if (abs(xloc-xerange(1,1,iie)).lt. d2chk) iitest(1) = 1
         if (abs(xloc-xerange(2,1,iie)).lt. d2chk) iitest(1) = 1
         if (abs(yloc-xerange(1,2,iie)).lt. d2chk) iitest(2) = 1
         if (abs(yloc-xerange(2,2,iie)).lt. d2chk) iitest(2) = 1
         if (abs(zloc-xerange(1,3,iie)).lt. d2chk) iitest(3) = 1
         if (abs(zloc-xerange(2,3,iie)).lt. d2chk) iitest(3) = 1

         iic = 0

         if (iitest(1) .eq. 1) then
         if (iitest(2) .eq. 1) then
         if (iitest(3) .eq. 1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc + sign(d2chk,rpart(jr,ip))
            pxyzg(2,iic) = yloc + sign(d2chk,rpart(jr+1,ip))
            pxyzg(3,iic) = zloc + sign(d2chk,rpart(jr+2,ip))
         endif
         endif
         endif

         if (iitest(1).eq.1) then
         if (iitest(2).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc + sign(d2chk,rpart(jr,ip))
            pxyzg(2,iic) = yloc + sign(d2chk,rpart(jr+1,ip))
            pxyzg(3,iic) = zloc
         endif
         endif

         if (iitest(1).eq.1) then
         if (iitest(3).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc + sign(d2chk,rpart(jr,ip))
            pxyzg(2,iic) = yloc
            pxyzg(3,iic) = zloc + sign(d2chk,rpart(jr+2,ip))
         endif
         endif

         if (iitest(2).eq.1) then
         if (iitest(3).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc
            pxyzg(2,iic) = yloc + sign(d2chk,rpart(jr+1,ip))
            pxyzg(3,iic) = zloc + sign(d2chk,rpart(jr+2,ip))
         endif
         endif

         if (iitest(1).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc + sign(d2chk,rpart(jr,ip))
            pxyzg(2,iic) = yloc
            pxyzg(3,iic) = zloc
         endif

         if (iitest(2).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc
            pxyzg(2,iic) = yloc + sign(d2chk,rpart(jr+1,ip))
            pxyzg(3,iic) = zloc
         endif

         if (iitest(3).eq.1) then
            iic = iic + 1

            pxyzg(1,iic) = xloc
            pxyzg(2,iic) = yloc
            pxyzg(3,iic) = zloc + sign(d2chk,rpart(jr+2,ip))
         endif

         do j= 1,iic
            xxloc = pxyzg(1,j)
            yyloc = pxyzg(2,j)
            zzloc = pxyzg(3,j)
            call bounds_p_check(xxloc,xdrange(1,1),xdrange(2,1),ifmovex)
            call bounds_p_check(yyloc,xdrange(1,2),xdrange(2,2),ifmovey)
            call bounds_p_check(zzloc,xdrange(1,3),xdrange(2,3),ifmovez)

            nfpts = nfpts + 1
            fptsmap(nfpts) = ip

            call copy (rfpts(1,nfpts),rpart(1,ip),lr)
            call icopy(ifpts(1,nfpts),ipart(1,ip),li)

            rfpts(jx,nfpts) = xxloc
            rfpts(jy,nfpts) = yyloc
            rfpts(jz,nfpts) = zzloc

            inpt(1,nfpts) = ifmovex
            inpt(2,nfpts) = ifmovey
            inpt(3,nfpts) = ifmovez
         enddo
      enddo

      call findpts(i_fp_hndl !  stride     !   call findpts( ihndl,
     $           , ifpts(jrc,1),li        !   $             rcode,1,
     $           , ifpts(jpt,1),li        !   &             proc,1,
     $           , ifpts(je0,1),li        !   &             elid,1,
     $           , rfpts(jr ,1),lr        !   &             rst,ndim,
     $           , rfpts(jd ,1),lr        !   &             dist,1,
     $           , rfpts(jx ,1),lr        !   &             pts(    1),1,
     $           , rfpts(jy ,1),lr        !   &             pts(  n+1),1,
     $           , rfpts(jz ,1),lr ,nfpts)    !   &             pts(2*n+1),1,n)


      jps = jpid1-1     ! Pointer to temporary proc id for swapping
      do i=1,nfpts
         ip = fptsmap(i)

         if (inpt(1,i) .eq.0) rfpts(jx,i) = rpart(jx,ip)
         if (inpt(2,i) .eq.0) rfpts(jy,i) = rpart(jy,ip)
         if (inpt(3,i) .eq.0) rfpts(jz,i) = rpart(jz,ip)

          if (inpt(1,i).eq.1) rfpts(jx,i) = rfpts(jx,i)  -
     >          sign(d2chk,rpart(jr,ip))
          if (inpt(2,i).eq.1) rfpts(jy,i) = rfpts(jy,i) -
     >        sign(d2chk,rpart(jr+1,ip))
          if (inpt(3,i).eq.1) rfpts(jz,i) = rfpts(jz,i) -
     >        sign(d2chk,rpart(jr+2,ip))

          ifpts(jpid,i) = 1
          if (inpt(1,i)+inpt(2,i)+inpt(3,i).lt.0.9) then
          if (ifpts(jpt,i) .eq. nid) then
             ifpts(jpid,i) = -1   ! flag if should not search
          endif
          endif

         ifpts(jps,i) = ifpts(jpt,i)
      enddo


c     Move particle info to the processor that owns each particle
c     using crystal router in log P time:
      call crystal_tuple_transfer(i_cr_hndl,nfpts,lpart
     $           , ifpts,ni,partl,nl,rfpts,nr,jps)


c     now loop over all the particles on this element
      do i = 1,n
         nneigh = 0
c        particles in local elements
         do j = 1,n
            if (i .ne. j) then
               pdist = abs(rpart(jx,i)-rpart(jx,j))**2
     >                          + abs(rpart(jy,i)-rpart(jy,j))**2
     >                          + abs(rpart(jz,i)-rpart(jz,j))**2
               pdist = sqrt(pdist)
               if (pdist .gt. d3) goto 1109
               nneigh = nneigh + 1
            endif
1109        continue
         enddo

c        particles in different elements
         do j = 1,nfpts
            if( ifpts(jpid1,j).eq.ipart(jpid1,i)) then
            if( ifpts(jpid2,j).eq.ipart(jpid2,i)) then
            if( ifpts(jpid3,j).eq.ipart(jpid3,i)) then
               goto 11092
            endif
            endif
            endif
            if (ifpts(jpid,j).gt. 0) then
            if (ipart(je0,i) .eq. ifpts(je0,j)) then
            pdist = abs(rpart(jx,i)-rfpts(jx,j))**2
     >                    + abs(rpart(jy,i)-rfpts(jy,j))**2
     >                    + abs(rpart(jz,i)-rfpts(jz,j))**2
            pdist = sqrt(pdist)
            if (pdist .gt. d3) goto 11092
            nneigh = nneigh + 1
            endif
            endif
11092       continue
         enddo
         ipart(jpnn,i) = nneigh
         ipart(jai,i) = ipart(jai,i) - ipart(jpnn,i)
      enddo





      return
      end
c-----------------------------------------------------------------------

      subroutine bounds_p_check_cmtbone(xx,xl,xr,ifmove)

      ifmove = 0
      if (xx .gt. xr) then
         xx = abs(xx - xr) + xl
         ifmove = 1
      endif
      if (xx .lt. xl) then
         xx = xr - abs(xx - xl)
         ifmove = 1
      endif

      return
      end

c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      subroutine interp_props_part_location_cmtbone
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'CMTDATA'
      parameter (lr=16*ldim,li=5+6)
      common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 16*ndim
      common  /cparti/ ipart(li,lpart) ! Minimal value of li = 5
      common  /iparti/ n,nr,ni

      common /fundpart/ rhs_fluidp(lx1,ly1,lz1,ldim,lelt)
     >                 ,lhs_density(lx1,ly1,lz1,lelt)
      real rhs_fluidp,lhs_density

      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid
      common /BARRYREP/ rep, bot
      real              rep(lx1,ly1,lz1), bot
      nxyz = nx1*ny1*nz1
c     print *, 'before interp_props_part_location, nid: ', nid, 'n:', n 
        do i=1,n
           rrdum = 1.0
           if(if3d) rrdum = rpart(jr+2,i)
c          print *, 'nid:',nid, 'rpart(jr):', rpart(jr,i), 
c    >        rpart(jr+1,i), rpart(jr+2,i)
c          print *, 'nid:',nid, 'rrdum:', rrdum
           call init_baryinterp(rpart(jr,i),rpart(jr+1,i),rrdum)
           ie  =  ipart(je0,i) + 1
           call baryinterp(vx(1,1,1,ie),rpart(ju0,i))
           call baryinterp(vy(1,1,1,ie),rpart(ju0+1,i))
           if (if3d) call baryinterp(vz(1,1,1,ie),rpart(ju0+2,i))
        enddo
c     print *, 'in interp_props_part_location, nid: ', nid, 'n:', n 
      return
      end
c----------------------------------------------------------------------

      subroutine interp_comm_part()

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common  /iparti/ n,nr,ni

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /myparth/ i_fp_hndl, i_cr_hndl

      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      integer gfirst, inoassignd, resetFindpts, pload

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      logical partl         ! This is a dummy placeholder, used in cr()
      nl = 0                ! No logicals exchanged
      if ((icalld1.eq.0) .or. (resetFindpts .eq. 1))then
         tolin = 1.e-12
         if (wdsize.eq.4) tolin = 1.e-6
         call intpts_setup  (tolin,i_fp_hndl)
         call crystal_setup (i_cr_hndl,nekcomm,np)
         icalld1 = icalld1 + 1
c        resetFindpts = 0
c        added by keke
         print *, 'before transfer nid: ', nid, '# particles: ', n
c        end added by keke
      endif

      end

c-----------------------------------------------------------------------
      subroutine interp_u_for_adv(rpart,nr,ipart,ni,n,ux,uy,uz)
c     Interpolate fluid velocity at current xyz points and move
c     data to the processor that owns the points.
c     Input:    n = number of points on this processor
c     Output:   n = number of points on this processor after the move
c     Code checks for n > lpart and will not move data if there
c     is insufficient room.
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      common /myparth/ i_fp_hndl, i_cr_hndl

      real    pt_timers(10), scrt_timers(10)
      common /trackingtime/ pt_timers, scrt_timers

      real    rpart(nr,n),ux(1),uy(1),uz(1)
      integer ipart(ni,n)

      parameter (lrf=4+ldim,lif=5+1)
      real               rfpts(lrf,lpart)
      common /fptspartr/ rfpts
      integer            ifpts(lif,lpart),fptsmap(lpart)
      common /fptsparti/ ifpts,fptsmap

      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
     >               ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
     >               ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      logical partl         ! This is a dummy placeholder, used in cr()
      nl = 0                ! No logicals exchanged

      scrt_timers(4) = dnekclock()
      call interp_comm_part()

      scrt_timers(9) = dnekclock()

!     find particles in this rank, put into map
      call particles_in_nid(fptsmap,rfpts,lrf,ifpts,lif,nfpts,rpart,nr
     $                     ,ipart,ni,n)

c     lif is a 'block' size to know how many indexes to skip on next
c     jrc, jpt, and je0 store the find results
      scrt_timers(5) = dnekclock()
      call findpts(i_fp_hndl !  stride     !   call findpts( ihndl,
     $           , ifpts(jrc,1),lif        !   $             rcode,1,
     $           , ifpts(jpt,1),lif        !   &             proc,1,
     $           , ifpts(je0,1),lif        !   &             elid,1,
     $           , rfpts(jr ,1),lrf        !   &             rst,ndim,
     $           , rfpts(jd ,1),lrf        !   &             dist,1,
     $           , rfpts(jx ,1),lrf        !   &             pts(    1),1,
     $           , rfpts(jy ,1),lrf        !   &             pts(  n+1),1,
     $           , rfpts(jz ,1),lrf ,nfpts)    !   &             pts(2*n+1),1,n)      scrt_timers(5) = dnekclock() - scrt_timers(5)
      pt_timers(5) = scrt_timers(5) + pt_timers(5)

      nmax = iglmax(n,1)
      if (nmax.gt.lpart) then
         if (nid.eq.0) write(6,1) nmax,lpart
    1    format('WARNING: Max number of particles:'
     $   i9,'.  Not moving because lpart =',i9,'.')
      else
         scrt_timers(6) = dnekclock()
!        copy rfpts and ifpts back into their repsected positions in rpart and ipart
         call update_findpts_info(rpart,nr,ipart,ni,n,rfpts,lrf
     $                       ,ifpts,lif,fptsmap,nfpts)
         scrt_timers(9) = dnekclock() - scrt_timers(9) - scrt_timers(5)
         pt_timers(9) = scrt_timers(9) + pt_timers(9)
!        Move particle info to the processor that owns each particle
!        using crystal router in log P time:

         jps = jai-1     ! Pointer to temporary proc id for swapping
         do i=1,n        ! Can't use jpt because it messes up particle info
            ipart(jps,i) = ipart(jpt,i)
         enddo

!        sends and receives particle information (updating n)
         call crystal_tuple_transfer(i_cr_hndl,n,lpart
     $              , ipart,ni,partl,nl,rpart,nr,jps)
!        Sort by element number - for improved local-eval performance
         call crystal_tuple_sort    (i_cr_hndl,n
     $              , ipart,ni,partl,nl,rpart,nr,je0,1)
         pt_timers(6) = pt_timers(6) + dnekclock() - scrt_timers(6)
      endif

!     Interpolate (locally, if data is resident).
      scrt_timers(7) = dnekclock()
!      call baryweights_findpts_eval(rpart,nr,ipart,ni,n)
      pt_timers(7) = pt_timers(7) + dnekclock() - scrt_timers(7)
      pt_timers(4) = pt_timers(4) + dnekclock() - scrt_timers(4)
      return
      end

