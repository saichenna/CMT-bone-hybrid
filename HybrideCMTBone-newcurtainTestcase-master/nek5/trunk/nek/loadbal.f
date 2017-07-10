      subroutine assign_partitions !(gllnid, lelt, nelgt, np)
c     This subroutine is used for update gllnid            
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL' !these include contains gllnid, lelt, and np     

      !integer gllnid(1) !, lelt, nelgt, np
      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      integer gfirst, inoassignd, resetFindpts
      integer pload
      integer psum(lelg)
      integer nw, k, nel, nmod, npp  !the number of particles
c     nw=200 
      
      call izero(pload, lelg)
c      call randGenet(pload, nelgt, nw) !random assign pload
c     call preSum(pload, psum, nelgt) !calculate the prefix sum of pload
c      call ldblce(psum, nelgt, gllnid, np) !equally distribute the load to np processor, assigned to gllnid
      
c     uniformally assign element
      nel = nelgt/np
      nmod  = mod(nelgt,np)  ! bounded between 1 ... np-1
      npp   = np - nmod      ! how many paritions of size nel
      ! setup partitions of size nel
      k   = 0
      do ip = 0,npp-1
         do e = 1,nel
            k = k + 1
            gllnid(k) = ip
         enddo
      enddo
      ! setup partitions of size nel+1
      if(nmod.gt.0) then
        do ip = npp,np-1
           do e = 1,nel+1
              k = k + 1
              gllnid(k) = ip
           enddo
        enddo
      endif

c     part = nelgt/np
c     do i=1, nelgt
c        gllnid(i) = (i-1)/part
c     enddo
c      call printi(gllnid, nelgt)

      end subroutine

c----------------------------------------------------------------

c------------------------------------------------------------------

c     subroutine to get prefix sum
      subroutine preSum(pload, psum, len)
      integer pload(len)
      integer psum(len)
      integer i
      psum(1)=pload(1)
      do 30 i=2, len
          psum(i)=psum(i-1)+pload(i)
  30  continue

c      do 50 i=1, len
c         pload(i)=psum(i)
c  50  continue

      return
      end

c--------------------------------------------------------
c     assign to corresponding processor, limit the number of element
c     assigned to each processor
      subroutine ldblce_limit(psum, len, gllnid, np)
         include "SIZE"
         integer psum(len)
         integer np
         integer gllnid(len)

         integer i,j, k, flag, ne
         integer pos(np+1)  !pos(i)-pos(i+1)-1 belong to processor i-1
         real diff, diffn, thresh
         i=1
         flag = 0
         thresh=psum(len)*1.0/np*i
         call izero(pos, np+1)
         pos(1)=1
         ne = 0
         do 70 j=2, len
            diff=abs(thresh-psum(j-1))
            diffn=abs(thresh-psum(j))
            if(diff .ge. diffn) then
               !write(*,*) "i:", i, "pos(i):", pos(i)
               ! bring in lelt
               ne=ne+1 
               if (ne > lelt-1) then
                   print *,i, "Number of elements",
     $              "exceeds lelt = ", lelt
                   pos(i+1)=j
                   ne = 0
                   i = i + 1
                   thresh=psum(len)*1.0/np*i
               else 
                   pos(i+1)=j+1
               endif
            else
               pos(i+1)=j
               !write(*,*) "i/:", i, "pos(i):", pos(i)
               i=i+1
               thresh=psum(len)*1.0/np*i
               ne = 0
            endif
  70      continue
c         print *, 'prefix sum, len: ', len
c         call printi(psum, len)
c         print *, ' i', i
          !call printi(pos, np+1)
          if( i .lt. np) then ! this part is for the partition less than np
              do k = i+2, np+1
                 pos(k) = len + 1
              enddo
          endif 
c         print *, 'printing pos'
c         call printi(pos, np+1)
c         do i=1, np+1  !verify loadbal
c            print *,'pos:', i, pos(i)
c         enddo 

c         print *, 'load of p', 0, psum(pos(2)-1)
c    $                 ,pos(1), pos(1+1)-1    
          do 80 i=1, np
c            print *, 'load of p', i-1, psum(pos(i+1)-1)-psum(pos(i)-1)
c    $                 ,pos(i), pos(i+1)-1    
             do 90 j=pos(i), pos(i+1)-1
                gllnid(j)=i-1
  90         continue
  80      continue
c         print *, 'printing gllnid, length: ',len 
c         call printi(gllnid, len)
          
      return
      end
c--------------------------------------------------------
c     assign to corresponding processor of gllnid, distributed method
      subroutine ldblce_dist_new(psum, newgllnid)
      include 'mpif.h'
      include 'SIZE'
      include 'PARALLEL'
      include 'CMTPART'

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal

      integer psum(nelt), newgllnid(nelgt)
      real ithreshb!, threshil, threshh
      integer totalload, diva, ierr, pnelt, m, j
      integer tempgllnid(nelt), preElem
      integer nelt_array_psum(np)
      integer status_mpi(MPI_STATUS_SIZE)
      integer posArray(2, np), npos, nposArray(np), posArrayGlo(2,np)
      integer preSum_npos, npos_array(np)
!     posArray store where the element change; for example, if
!     tempgllnid={0,0,0,1,1,1,2,3,3}, posArray=
!     (0,1)(1,4)(2,7)(3,8)

      totalload = nw + ceiling(nw*1.0/nelgt)*nelgt
      !print *, nid_, totalload

      threshb = totalload*1.0/np

      do i = 1, nelt
         diva = AINT((psum(i)-1)*1.0/threshb)
         tempgllnid(i) = diva
      enddo

c     send tempgllnid(nelt) to the next processor, except for the last processor
      if(nid .ne. np-1) then
         call mpi_send(tempgllnid(nelt), 1, mpi_integer, nid+1, 0
     $      , nekcomm, ierr)
c        print *, 'send ', nid, nid+1, tempgllnid(nelt)
      endif

      if(nid .ne. 0) then
         call mpi_recv(preElem, 1, mpi_integer, nid-1, 0, nekcomm
     $      , status_mpi, ierr)
c        print *, 'receive ', nid-1, nid, preElem
      endif

c     Get exclusive prefix sum of nelt, sort it in pnelt
      pnelt = igl_running_sum_ex(nelt)

      call izero(posArray, 2*np)
      npos = 0
      if(nid .eq. 0) preElem = -1
      if(preElem .ne. tempgllnid(1)) then
         npos = npos + 1
         posArray(1,npos) = tempgllnid(1)
         posArray(2,npos) = 1 + pnelt !get the local element id
      endif
      preElem = tempgllnid(1)
      do i=2, nelt
          if(preElem .ne. tempgllnid(i)) then
             npos = npos + 1
             posArray(1,npos) = tempgllnid(i)
             posArray(2,npos) = i + pnelt !get the local element id
             preElem = tempgllnid(i)
          endif
      enddo

c     do i= 1, npos
c        print *, 'ElmProBegin', nid, npos, posArray(1,i),
c    $       posArray(2,i)
c     enddo

c     Get exclusive prefix sum of npos, store it in preSum_npos
c     npos   1 2 1 1  (npos_array: number of cut/transition points in processors 
c                       P1, P2, P3, P4) 
c     preSum_npos 0 1 3 4 
c     nposArray   0 1 3 4 in every processor (start position of npos in
c                                             global buffer)
c     print *, 'pos:', nid, npos
      npos = 2*npos !*2 because posArray is 2-d array
      preSum_npos = igl_running_sum_ex(npos)
c     print *, nid, preSum_npos
      call mpi_allgather(preSum_npos, 1, mpi_integer, nposArray
     $    , 1,  MPI_INTEGER, nekcomm, ierr)

c     Get nelt_array for every processor
      call mpi_allgather(npos, 1, mpi_integer, npos_array, 1,
     $   MPI_INTEGER, nekcomm, ierr) !note here, the receive buff is also 1 not np

c     Gather posArray into posArrayGlo
      call mpi_allgatherv(posArray, npos, mpi_integer, posArrayGlo,
     $   npos_array, nposArray, mpi_integer, nekcomm, ierr)

c     do i=1,np
c        print *, 'posArrayGlo', nid, i, posArrayGlo(1,i)
c    $      , posArrayGlo(2,i)
c     enddo


c     limit the number of element in a processor
      do i = 2, np
         if(posArrayGlo(2,i)-posArrayGlo(2,i-1) .gt. lelt) then
            if(nid .eq. 0)
     $      print *, 'proc ', i-1, 'exceed lelt, rearranged'
            posArrayGlo(2,i) = posArrayGlo(2,i-1)+lelt
         endif
      enddo
      if(nelgt+1-posArrayGlo(2,np) .gt. lelt) then
         posArrayGlo(2,np) = nelgt+1-lelt
         do i=np, 2, -1
            if(posArrayGlo(2,i)-posArrayGlo(2,i-1) .gt. lelt) then
               if(nid .eq. 0)
     $           print *, 'proc ', i-1, 'exceed lelt, rearranged'
               posArrayGlo(2,i-1) = posArrayGlo(2,i)-lelt
            else
               exit
            endif
         enddo
      endif
c     print *, 'load of p', i-1, psum(pos(i+1)-1)-psum(pos(i)-1)
c    $                 ,pos(i), pos(i+1)-1
      do i=1,np-1
         do j=posArrayGlo(2,i),  posArrayGlo(2,i+1)-1
            newgllnid(j)=posArrayGlo(1,i)
         enddo
      enddo
      do i = j, nelgt
         newgllnid(i) = np-1
      enddo

c     do i =1, nelgt
c        print *, 'newgllnid', nid, i, newgllnid(i)
c     enddo

      return
      end


c--------------------------------------------------------
c-----------------------------------------------------------------

c      print array real
       subroutine printr(pload, len)
          real pload(len)
          integer i
          do 40 i=1, len
             print *, pload(i)
   40     continue
       return
       end

c      print array integer
       subroutine printi(pos, len)
          integer pos(len)
          integer i
          do 40 i=1, len
             print *, pos(i)
   40     continue
       return
       end


c------------------------------------------------------------------
c     recompute partitions
       subroutine recompute_partitions_distr
          include 'SIZE'
          include 'INPUT'
          include 'PARALLEL'          
          include 'TSTEP'          
          include 'SOLN'          
          include 'CMTPART'          
          include 'CMTDATA'          
 
          !parameter (lr=16*ldim,li=5+6)
c         parameter (lr=76,li=10)
          common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
          common /elementload/ gfirst, inoassignd, 
     >                 resetFindpts, pload(lelg)
          integer gfirst, inoassignd, resetFindpts, pload

c         integer nw
c         common /particlenumber/ nw

          integer newgllnid(lelg), trans(3, lelg), trans_n, psum(lelt)
c          integer total
          integer e,eg,eg0,eg1,mdw,ndw
          integer ntuple, i, el, delta, nxyz
c     common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jpid
c    >                   ,jai,nai,    jr,jd,jx,jy,jz,jv0,ju0,jf0,jfusr
c    >                   ,jfqs,jfun,jfiu,jtaup,jcd,jdrhodt,jre,jDuDt
c    >                   ,jtemp,jrho,jrhop,ja,jvol,jdp,jar,jx1,jx2,jx3
c    >                   ,jv1,jv2,jv3,ju1,ju2,ju3,nar,jvol1,jgam

c         common  /cparti/ ipart(li,llpart)
          real   xerange(2,3,lelt)
          common /elementrange/ xerange

c         common  /iparti/ n,nr,ni
c         integer particleMap(3, lelg)
          integer gaselement(np), ierr, nn, partialload, totalload
          real ratio

          if(nid_ .eq. 0) then
             print *, 'in recompute_partitions_distr'
          endif
          call nekgsync()          
          starttime = dnekclock()
          nxyz = nx1*ny1*nz1   !total # of grid point per element
          call izero(pload, lelg)
          delta = ceiling(nw*1.0/nelgt)
          ratio = 1.0
c         if(nid_ .eq. 0) then
c            print *, 'in recompute_partitions_distr'
c            print * ,'nw:', nw, 'delta:', delta, 'ratio', ratio
c         endif
          ntuple = nelt
c         do i=1,ntuple
c            eg = lglel(i)
c            particleMap(1,i) = eg
c            particleMap(2,i) = 0        ! processor id to send for element eg
c            particleMap(3, i) = 0      !  #of particles in each element, reinitialize to 0, otherwise it keeps the previous value
c         enddo

          call izero(pload, lelg) 
          do ip=1,n
             el = ipart(je0, ip) + 1      ! element id in ipart is start from 0, so add 1
c            particleMap(3, el) = particleMap(3, el) + 1
             pload(el) = pload(el)+1
          enddo

          do i=1,ntuple
c                particleMap(3, i) = particleMap(3, i) + delta*ratio
                 pload(i) = pload(i) + delta*ratio
          enddo
c          do i=1,ntuple
c             pload(i) = particleMap(3, i)
c          enddo

           call izero(psum, lelt)
           call preSum(pload, psum, nelt)

c          t0 = dnekclock()
           nn = psum(nelt)
           partialload= igl_running_sum_ex(nn) !exclusive prefix sum
c         if( mod(nid, np/2) .eq. np/2-2 .or. nid .eq.0) then
c         print *, " igl_running_sum_ex" , dnekclock() - t0, t0,
c    $        dnekclock()
c         endif

c          now every processor has the actual prefix sum
           do i=1, nelt
              psum(i) = psum(i) + partialload
           enddo
           !print *, nid, nelt, n, 'psum(nelt): ', psum(nelt)
           !if (nid .eq.0) call printi(gllel, lelt) !printi(lglel,lelt)
           !print *, 'finish'
           !if (nid .eq.0) call printi(lglel,lelt)

           call izero(newgllnid, lelg)
           if( mod(nid, np/2) .eq. np/2-2) then
          print *, "before ldblce_dist_new" , dnekclock() - starttime
          endif
c          call ldblce_dist(psum, newgllnid)
           t0 = dnekclock()
           call ldblce_dist_new(psum, newgllnid)
           if( mod(nid, np/2) .eq. np/2-2) then
          print *, "in ldblce_dist_new" , dnekclock() - t0
          endif
           t0 = dnekclock()

          call izero(trans, 3*lelg)
          call track_elements(gllnid, newgllnid, nelgt, trans,
     $                               trans_n, lglel)
c         do i=1, trans_n
c           print *, nid_, 'trans_n', trans_n,
c    $           trans(1, i), trans(2, i), trans(3, i)
c         enddo

          call track_particles(trans, trans_n)
          call nekgsync()
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
          print *, "Component 0:" , endtime - starttime!, starttime,
c    $         endtime
          print *, "after ldblce_dist_new:" , endtime - t0
          endif
          !call mergePhigArray(newgllnid, trans, trans_n)
          !call mergeUArray(newgllnid)
          !call mergeTlagArray(newgllnid)
          call mergeArray(phig, lx1*ly1*lz1, lelt, newgllnid, 1)
c         print *, 'complete merge phig', nid
          call mergeArray(u, lx1*ly1*lz1*toteq, lelt, newgllnid, 2)
c         print *, 'complete merge u', nid
c         call mergeUArray(newgllnid)
c         call mergeTlagArray(newgllnid)
          call mergeTlagArray_new(newgllnid)
c         print *, 'complete merge tlag', nid

          call icopy(gllnid, newgllnid, nelgt)
              
          return
          end
c------------------------------------------------------------------
c     recompute partitions
       !subroutine recompute_partitions
       subroutine recompute_partitions_cpu
          include 'SIZE'
          include 'INPUT'
          include 'PARALLEL'          
          include 'TSTEP'          
          include 'SOLN'          
          include 'CMTPART'          
          include 'CMTDATA'          
 
c         parameter (lr=76,li=10)
          common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
          common /elementload/ gfirst, inoassignd, 
     >                 resetFindpts, pload(lelg)
          integer gfirst, inoassignd, resetFindpts, pload

c         integer nw
c         common /particlenumber/ nw

          integer newgllnid(lelg), trans(3, lelg), trans_n, psum(lelg)
          integer e,eg,eg0,eg1,mdw,ndw
          integer ntuple, i, el, delta, nxyz
c     common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jpid
c    >                   ,jai,nai,    jr,jd,jx,jy,jz,jv0,ju0,jf0,jfusr
c    >                   ,jfqs,jfun,jfiu,jtaup,jcd,jdrhodt,jre,jDuDt
c    >                   ,jtemp,jrho,jrhop,ja,jvol,jdp,jar,jx1,jx2,jx3
c    >                   ,jv1,jv2,jv3,ju1,ju2,ju3,nar,jvol1,jgam

c         common  /cparti/ ipart(li,llpart)
          real   xerange(2,3,lelt)
          common /elementrange/ xerange

c         common  /iparti/ n,nr,ni
          integer particleMap(3, lelg)
          integer gaselement(np)
          real ratio
          
c         if(nid .eq. 0) then
c             print *, 'old pload:'
c             call printi(pload, nelgt)
c         endif
          call nekgsync()
          starttime = dnekclock()
          nxyz = nx1*ny1*nz1   !total # of grid point per element
          delta = ceiling(nw*1.0/nelgt)
          ratio = 1.0
          ntuple = nelt
c         if(nid .eq. 0) then
c            print *, 'delta:', delta, 'nw:', nw, 'ntuple:', ntuple
c            print *, 'starttime', starttime 
c         endif
          do i=1,ntuple
             eg = lglel(i)
             particleMap(1,i) = eg
             particleMap(2,i) = 0        ! processor id to send for element eg
             particleMap(3, i) = 0      !  #of particles in each element, reinitialize to 0, otherwise it keeps the previous value
          enddo

          do ip=1,n
             el = ipart(je0, ip) + 1      ! element id in ipart is start from 0, so add 1
c            print *, 'ipart, nid', istep,  nid, ip, el, ntuple
             particleMap(3, el) = particleMap(3, el) + 1
          enddo

c         gas_right_boundary = exp(TIME/2.0)



          do i=1,ntuple
c            x_left_boundary = xerange(1,1,i)
c            if (x_left_boundary .lt. gas_right_boundary) then
c            !if (vx(1,1,1,i) .ne. 0) then
c               print *, 'istep, nid',istep,  nid, 'particle map', 
c    $            i, particleMap(1,i)
c    $           ,particleMap(2,i),particleMap(3,i)
                particleMap(3, i) = particleMap(3, i) + delta*ratio
c            else
c               particleMap(3, i) = 0
c            endif
          enddo
          mdw=3
          ndw=nelgt
          key = 2  ! processor id is in wk(2,:)
          call crystal_ituple_transfer(cr_h,particleMap,
     $                                 mdw,ntuple,ndw,key)
          
c          total=lelt*10
          
          if (nid .eq. 0) then
             key=1
             nkey = 1
             call crystal_ituple_sort(cr_h,particleMap,mdw,
     $                                ntuple,key,nkey)
             do i=1,ntuple
                pload(i) = particleMap(3, i)
             enddo

c            print *, 'new pload:'
c            call printi(pload, nelgt)
             !print *, 'new pload/n'
             !call printr(newPload, lelt)
             call izero(psum, lelg)
             call preSum(pload, psum, nelgt)
             print *, 'recompute_partitions: psum(nelgt): ', psum(nelgt)
             print *, 'ratio:', ratio
             !call printr(newPload, lelt)
             
c            do i=1, nelgt
c               print *, 'psum', i, psum(i)
c            enddo 
             call izero(newgllnid, lelg) 
             print *, "before ldblce_limit", dnekclock()-starttime
c    $            ,starttime, dnekclock()
             t0 = dnekclock()  
             call ldblce_limit(psum, nelgt, newgllnid, np_)
             print *, "in ldblce_limit", dnekclock()-t0
             t0 = dnekclock()  
c            do i=1, nelgt
c               print *, 'newgllnid', i, newgllnid(i)
c            enddo 

c            print *, 'print new gllnid'
c            call printi(newgllnid, nelgt)
c            call izero(gaselement, np)
c            do i=1, nelgt
c               if (pload(i) .ne. 0) then
c               gaselement(newgllnid(i)+1)=gaselement(newgllnid(i)+1)+1
c               endif
c            enddo 
c            do i=1, np
c                print *, '# gas element on', i-1, 'is: ', gaselement(i)
c            enddo
          endif
          call bcast(newgllnid,4*nelgt)

          call izero(trans, 3*lelg)
          call track_elements(gllnid, newgllnid, nelgt, trans, 
     $                               trans_n, lglel)
c         print *, 'print trans'
c         do 110 i=1, trans_n
c           print *, 'trans', trans(1, i), trans(2, i), trans(3, i)
c 110  continue

          call track_particles(trans, trans_n)
          call nekgsync()
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2 .or. nid .eq. 0) then
          print *, "Component 0:" , endtime - starttime !, starttime,
c    $                     endtime 
          endif
          if( nid .eq.0) then
          print *, "after ldblce_limit:" , endtime - t0 
          endif
          call mergeArray(phig, lx1*ly1*lz1, lelt, newgllnid, 1)
c         print *, 'complete merge phig', nid
          call mergeArray(u, lx1*ly1*lz1*toteq, lelt, newgllnid, 2)
c         print *, 'complete merge u', nid
c         call mergeTlagArray_new(newgllnid)
c         print *, 'complete merge tlag', nid

          call icopy(gllnid, newgllnid, nelgt)
c         print *, 'did icopy', nid

          return
          end
          
c------------------------------------------------------
c subroutine of track elements to be send and received
       subroutine track_elements(gllnid, newgllnid, len, trans, 
     $                     trans_n, lglel)
       include 'SIZE'
       integer gllnid(len), newgllnid(len), trans(3, len), trans_n
       integer lglel(1) !lglel(1)?
       !trans: first column stores the source pid, second column stores the target pid, the third column stores the element id
       integer i, j; !local variable

       trans_n=1;
       j=0;
       do i=1, len
          if ((gllnid(i) .eq. nid)) then
             j = j+1;
             if(gllnid(i) .ne. newgllnid(i)) then
c since added gllnid(i) .eq. nid, right now, the trans in processor i only store the elements that he shold send. Not all the processors.            
             trans(1, trans_n) = gllnid(i);
             trans(2, trans_n) = newgllnid(i);
             trans(3, trans_n) = lglel(j)
             trans_n=trans_n+1;
           endif
         endif
       enddo
       trans_n=trans_n-1;  !the length of trans
c      print *, trans_n, 'print again in track elements'
c       do 110 i=1, trans_n
          !do 120 j=1, width
c           print *, i, trans(i,1), trans(i,2), trans(i,3)
c  120     continue
c  110  continue

       return
       end
 
c-----------------------------------------------------
c update the particles that are in the elements to be
c transferred, and set jpt to the destination processor
       subroutine track_particles(trans, trans_n)
       include 'SIZE'
       include 'PARALLEL' 
       include 'CMTPART' 

       integer trans(3, lelg), trans_n
C      parameter (lr=76,li=10)
c      common  /iparti/ n,nr,ni
c      common  /cpartr/ rpart(lr,llpart) ! Minimal value of lr = 16*ndim
c      common  /cparti/ ipart(li,llpart) ! Minimal value of li = 5
c      common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
c    >                ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
c    >                ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid
c     common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jpid
c    >                   ,jai,nai,    jr,jd,jx,jy,jz,jv0,ju0,jf0,jfusr
c    >                   ,jfqs,jfun,jfiu,jtaup,jcd,jdrhodt,jre,jDuDt
c    >                   ,jtemp,jrho,jrhop,ja,jvol,jdp,jar,jx1,jx2,jx3
c    >                   ,jv1,jv2,jv3,ju1,ju2,ju3,nar,jvol1,jgam

       common /myparth/ i_fp_hndl, i_cr_hndl
     
       
       integer ip, it, e
       logical partl         ! This is a dummy placeholder, used in cr()
       nl = 0                ! No logicals exchanged

       
c     change ipart(je0,i) to global element id
       ip=0
       do ip = 1, n
           e = ipart(je0, ip) + 1 ! je0 start from 0
           ipart(je0, ip) = lglel(e)
       enddo

       do ip = 1, n
          !e = ipart(je0,ip)
          do it = 1, trans_n
             if(ipart(je0, ip) .eq. trans(3, it)) then  
                ipart(jpt, ip) = trans(2, it) !new processor
                exit
             endif
          enddo
          ipart(jps, ip) = ipart(jpt, ip)
       enddo 

       call crystal_tuple_transfer(i_cr_hndl,n,llpart
     $              , ipart,ni,partl,nl,rpart,nr,jps)


       return 
       end 
c------------------------------------------------------
c subroutine to merge  arrays
          subroutine mergeArray(arrayName, len1, len2, newgllnid, atype)
          include 'SIZE'
          include 'INPUT'
          include 'PARALLEL'          
          include 'CMTDATA'          

          integer newgllnid(lelg), trans(3, lelg), len1, len2
          real arrayName(len1, len2)
          real tempArray(len1, len2) 
          logical partl
          integer nl, sid, eid
          integer key, nkey, trans_n, nxyz
          integer atype

          starttime = dnekclock()
          trans_n=0
          nxyz = len1
          nl=0
          do i=1, nelt
              trans_n = trans_n +1
              ieg = lglel(i)
              if (gllnid(ieg) .eq. nid) then 
                  trans(1, trans_n) = gllnid(ieg)
                  trans(2, trans_n) = newgllnid(ieg)
                  trans(3, trans_n) = ieg
                  do l=1, nxyz
                     tempArray(l, trans_n) = arrayName(l,i)
                  enddo
              endif
          enddo
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
             if (atype .eq. 1) then
                print *, "Component 1b:" , endtime-starttime
             else
                print *, "Component 1c:" , endtime-starttime
             endif 
          endif 
c          print *, index_n, trans_n, lelt, nid
           
          starttime = dnekclock()
          key=2
          call crystal_tuple_transfer(cr_h, trans_n, nelgt, trans,
     $                    3, partl, nl, tempArray,nxyz,key)
c         print *, 'nid: ', nid, 'trans_n', trans_n      
          key=3
          nkey=1
          call crystal_tuple_sort(cr_h, trans_n, trans, 3, 
     $               partl, nl, tempArray, nxyz, key,nkey)
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
             if (atype .eq. 1) then
                print *, "Component 2b:" , endtime-starttime
             else
                print *, "Component 2c:" , endtime-starttime
             endif 
          endif

          !Update u
          ! set sid and eid
          starttime = dnekclock()
          !copy tempu to u
              do i=1, trans_n
                  do l=1, nxyz
                     arrayName(l,i) = tempArray(l, i)
                  enddo
              enddo
c      print *, "Update u array: u_n", u_n
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
             if (atype .eq. 1) then
                print *, "Component 3b:" , endtime-starttime
             else
                print *, "Component 3c:" , endtime-starttime
             endif 
          !print *, "nelgt", nelgt, mod(nid, nelgt/2), nelgt/2-2
          endif
       end
c------------------------------------------------------
c----------------------------------------------------------------------
c subroutine to merge tlag array
          subroutine mergeTlagArray_new(newgllnid)
          include 'SIZE'
          include 'INPUT'
          include 'PARALLEL'          
          include 'CMTDATA'          
          include 'SOLN'          

          integer newgllnid(lelg), trans(3, lelg)
          real tlagarray(lx1*ly1*lz1*(lorder-1)*ldimt, lelt) 
c         integer procarray(3, lelg) !keke changed real to integer to use crystal_ituple_transfer
c         real temptlag(lx1*ly1*lz1*(lorder-1)*ldimt, lelt)
          logical partl
          integer nl, sid, eid
          integer key, nkey, ifirstelement, ilastelement, u_n, trans_n

          starttime = dnekclock()
          trans_n=0
          nxyz = lx1*ly1*lz1*(lorder-1)*ldimt
          nl=0
          do i=1, nelt
              trans_n = trans_n + 1
              ieg = lglel(i)
              if (gllnid(ieg) .eq. nid) then
c                 procarray(1, index_n) = gllnid(ieg)
c                 procarray(2, index_n) = newgllnid(ieg)
c                 procarray(3, index_n) = ieg
                  trans(1, trans_n) = gllnid(ieg)
                  trans(2, trans_n) = newgllnid(ieg)
                  trans(3, trans_n) = ieg
                  do l=1, ldimt
                  do j=1, lorder-1
                      do k=1, lz1
                          do n=1, ly1
                             do m=1, lx1
                                ind=m+(n-1)*lx1+(k-1)*lx1*ly1+
     $                               (j-1)*lz1*ly1*lx1+
     $                               (l-1)*(lorder-1)*lz1*ly1*lx1 
                             tlagarray(ind, trans_n) = tlag(m,n,k,i,j,l)
                             enddo
                          enddo
                      enddo
                  enddo
                  enddo
              endif
          enddo
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
          print *, "Component 1a:" , endtime-starttime
          endif 
c          print *, index_n, trans_n, lelt, nid
           
          starttime = dnekclock()
          key=2
          call crystal_tuple_transfer(cr_h, trans_n, nelgt, trans,
     $                    3, partl, nl, tlagarray,nxyz,key)
c         print *, 'nid: ', nid, 'trans_n', trans_n      
          key=3
          nkey=1
          call crystal_tuple_sort(cr_h, trans_n, trans, 3, 
     $               partl, nl, tlagarray, nxyz, key,nkey)
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
          print *, "Component 2a:" , endtime-starttime
          endif

          !Update u
          ! set sid and eid
          starttime = dnekclock()
          !copy tempu to u
              do i=1, trans_n
                  do l=1, ldimt
                  do j=1, lorder-1
                      do k=1, lz1
                          do n=1, ly1
                             do m=1, lx1
                                ind=m+(n-1)*lx1+(k-1)*lx1*ly1+
     $                               (j-1)*lz1*ly1*lx1+
     $                               (l-1)*(lorder-1)*lz1*ly1*lx1
                            tlag(m,n,k,i,j,l) = tlagarray(ind, i)
                             enddo
                          enddo
                      enddo
                  enddo
                  enddo
              enddo
c      print *, "Update u array: u_n", u_n
          endtime = dnekclock()
          if( mod(nid, np/2) .eq. np/2-2) then
          print *, "Component 3a:" , endtime-starttime
          !print *, "nelgt", nelgt, mod(nid, nelgt/2), nelgt/2-2
          endif
       end
c----------------------------------------------------------------------
      subroutine update_ipartje0_to_local_cpu
      include 'SIZE'
      include 'PARALLEL'
      include 'CMTPART'
c     parameter (lr=16*ldim,li=5+6)
c     common  /iparti/ n,nr,ni
c     common  /cpartr/ rpart(lr,lpart) ! Minimal value of lr = 14*ndim+1
c     common  /cparti/ ipart(li,lpart) ! Minimal value of lr = 14*ndim+1
c     common /ptpointers/ jrc,jpt,je0,jps,jpid1,jpid2,jpid3,jpnn,jai
c    >                ,nai,jr,jd,jx,jy,jz,jx1,jx2,jx3,jv0,jv1,jv2,jv3
c    >                ,ju0,ju1,ju2,ju3,jf0,jar,jaa,jab,jac,jad,nar,jpid
      common /myparth/ i_fp_hndl, i_cr_hndl
      integer ip, e
      logical partl         ! This is a dummy placeholder, used in cr()
      nl = 0

      ip=0
      do ip = 1, n
         e = ipart(je0, ip)
         ipart(je0, ip) = gllel(e) - 1  ! je0 start from 0
      enddo
c     Sort by element number
      call crystal_tuple_sort(i_cr_hndl,n
     $          , ipart,ni,partl,nl,rpart,nr,je0,1)

      end
c----------------------------------------------------------------------
      subroutine reinitialize
      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'ZPER'
c
      include 'OPCTR'
      include 'CTIMER'

      logical ifemati,ifsync_
      common /elementload/ gfirst, inoassignd, resetFindpts, pload(lelg)
      integer gfirst, inoassignd, resetFindpts, pload
      real starttime, endtime

      inoassignd = 0
      call nekgsync()          
      starttime = dnekclock()
      etime = dnekclock()
      call readat
      etims0 = dnekclock_sync()
      if (nio.eq.0) then
         write(6,12) 'called reinitialization'
         write(6,12) 'nelgt/nelgv/lelt:',nelgt,nelgv,lelt
         write(6,12) 'lx1  /lx2  /lx3 :',lx1,lx2,lx3
         write(6,'(A,g13.5,A,/)')  ' done :: read .rea file ',
     &                             etims0-etime,' sec'
 12      format(1X,A,4I12,/,/)
      endif

      ifsync_ = ifsync
      ifsync = .true.

      call setvar          ! Initialize most variables !skip 

!#ifdef MOAB
!      if (ifmoab) call nekMOAB_bcs  !   Map BCs
!#endif

      instep=1             ! Check for zero steps
      if (nsteps.eq.0 .and. fintim.eq.0.) instep=0

      igeom = 2
      call setup_topo      ! Setup domain topology  

      call genwz           ! Compute GLL points, weights, etc.

      call io_init         ! Initalize io unit

!      if (ifcvode.and.nsteps.gt.0)
!     $   call cv_setsize(0,nfield) !Set size for CVODE solver

      if(nio.eq.0) write(6,*) 'call usrdat'
      call usrdat
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat'

      call gengeom(igeom)  ! Generate geometry, after usrdat 

      if (ifmvbd) call setup_mesh_dssum ! Set mesh dssum (needs geom)

      if(nio.eq.0) write(6,*) 'call usrdat2'
      call usrdat2
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat2'

      call geom_reset(1)    ! recompute Jacobians, etc.

      call vrdsmsh          ! verify mesh topology

!      call echopar ! echo back the parameter stack
      call setlog  ! Initalize logical flags

      call bcmask  ! Set BC masks for Dirichlet boundaries.

      if (fintim.ne.0.0.or.nsteps.ne.0)
     $   call geneig(igeom) ! eigvals for tolerances

      call vrdsmsh     !     Verify mesh topology

      call dg_setup    !     Setup DG, if dg flag is set.

      !starttime = dnekclock() 
      !if (ifflow.and.(fintim.ne.0.or.nsteps.ne.0)) then    ! Pressure solver 
         !call estrat                                       ! initialization.
         !print *, "estrat"
         !if (iftran.and.solver_type.eq.'itr') then         ! Uses SOLN space 
            !call set_overlap                               ! as scratch!
            !print *, "set_overlap"
         !elseif (solver_type.eq.'fdm'.or.solver_type.eq.'pdm')then
            !ifemati = .true.
            !kwave2  = 0.0
            !if (ifsplit) ifemati = .false.
            !call gfdm_init(nx2,ny2,nz2,ifemati,kwave2)
            !print *, "gfdm_init"
         !elseif (solver_type.eq.'25D') then
            !call g25d_init
            !print *, "g25d_init"
         !endif
      !endif
      !endtime = dnekclock()
      !if( mod(nid, np/2) .eq. np/2-2) then
          !print *, "pressure solver init", endtime - starttime
      !endif

!      call init_plugin !     Initialize optional plugin
      if(ifcvode) call cv_setsize

      if(nio.eq.0) write(6,*) 'call usrdat3'
      call usrdat3
      if(nio.eq.0) write(6,'(A,/)') ' done :: usrdat3'

#ifdef CMTNEK
        call nek_cmt_init
        if (nio.eq.0) write(6,*)'Initialized DG machinery'
#endif

      call nekgsync()          
      endtime = dnekclock()
      if( mod(nid, np/2) .eq. np/2-2) then
          print *, "total reinit time", endtime - starttime
      endif

c     call cmt_switch          ! Check if compiled with cmt
c     if (ifcmt) then          ! Initialize CMT branch
c       call nek_cmt_init
c       if (nio.eq.0) write(6,*)'Initialized DG machinery'
c     endif

c         for verification !!!!! 
c         call outpost(vx, vy, vz, pr, t, 'xyz')
c         call copy(vx, phig, nxyz*nelt)
c         for verification !!!!! 
       !endtime = dnekclock()
       !if( mod(nid, np/2) .eq. np/2-2) then
       !print *, "total reinit time", endtime - starttime
       !endif
       end

c------------------------------------------------------------------------
c------------------------------------------------------------------------
        subroutine printVerify
            include 'SIZE'
            common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
            
            print *, 'nid: ', nid_, 'nelt: ', nelt
        end

c--------------------------------------------------------------------
c-----------------------------------------------------------------------
      subroutine printTotalHeleq(e, eq)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn, e, eq
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3
  
      fmt = '(I4.4)'
      isprint = 0
      geid = e

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 3 .and. istep .le. 7) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      !do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
            write(x3, fmt) eq
          OPEN(UNIT=8800+eq+(geid-1)*5,FILE='harreq'//'.'//'id.'
     $        //trim(x1)//'.'
     $        //'step.'//trim(x2)//'.eqn.'//trim(x3),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=8800+eq+(geid-1)*5, FMT=*) m, n, k, eq
                       enddo
                   enddo
                enddo
            CLOSE(UNIT=8800+eq+(geid-1)*5)
            isprint = 0
          endif
        !enddo
        endif
      end


c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      subroutine printRes1arrayeleq(e, eq)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn, e, eq
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3
  
      fmt = '(I4.4)'
      isprint = 0
      geid = e

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 3 .and. istep .le. 7) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      !do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
            write(x3, fmt) eq
          OPEN(UNIT=8800+eq+(geid-1)*5,FILE='resarreq'//'.'//'id.'
     $        //trim(x1)//'.'
     $        //'step.'//trim(x2)//'.eqn.'//trim(x3),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=8800+eq+(geid-1)*5, FMT=*) m, n, k, eq, 
     $                 res1(m,n,k,pn,eq)
                       enddo
                   enddo
                enddo
            CLOSE(UNIT=8800+eq+(geid-1)*5)
            isprint = 0
          endif
        !enddo
        endif
      end


c-----------------------------------------------------------------------
      subroutine printRes1arrayel(e)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn, e
      character (len=8):: fmt !format descriptor
      character(5) x1, x2
  
      fmt = '(I4.4)'
      isprint = 0
      geid = e

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 3 .and. istep .le. 7) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      !do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
          OPEN(UNIT=9000+geid,FILE='resarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
              do i=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=9000+geid, FMT=*) m, n, k, i, res1(m,n,k,pn,i)
                       enddo
                   enddo
                enddo
             enddo
            CLOSE(UNIT=9000+geid)
            isprint = 0
          endif
        !enddo
        endif
      end


c-----------------------------------------------------------------------
      subroutine printRes1array
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2
  
      fmt = '(I4.4)'
      isprint = 0
      geid = 1

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 2500 .and. istep .le. 2600) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
          OPEN(UNIT=9000+geid,FILE='resarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
              do i=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=9000+geid, FMT=*) m, n, k, i, res1(m,n,k,pn,i)
                       enddo
                   enddo
                enddo
             enddo
            CLOSE(UNIT=9000+geid)
            isprint = 0
          endif
        enddo
        endif
      end


c-----------------------------------------------------------------------
      subroutine printGraduarray(e)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'
      include 'GEOM'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn, ind, e
      character (len=8):: fmt !format descriptor
      character(5) x1, x2
  
      fmt = '(I4.4)'
      isprint = 0
      geid = e

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 3 .and. istep .le. 7) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      !do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
          OPEN(UNIT=9200+geid,FILE='grarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
              do i=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
                          ind = m + (n-1)*lx1 + (k-1)*lx1*ly1
            WRITE(UNIT=9200+geid, FMT=*) gradu(ind,i,1), gradu(ind,i,2),
     $                          gradu(ind,i,3)
                       enddo
                   enddo
                enddo
              enddo
            CLOSE(UNIT=9200+geid)
            isprint = 0
          endif
        !enddo
        endif
      end


c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      subroutine printJacmiarray
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'
      include 'GEOM'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn, ind
      character (len=8):: fmt !format descriptor
      character(5) x1, x2
  
      fmt = '(I4.4)'
      isprint = 0
      geid = 1

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if(istep .ge. 3 .and. istep .le. 7) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            write(x1, fmt) geid
            write(x2, fmt) istep
          OPEN(UNIT=9100+geid,FILE='jacarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
                          ind = m + (n-1)*lx1 + (k-1)*lx1*ly1
            WRITE(UNIT=9100+geid, FMT=*) m, n, k, jacmi(ind,pn)
                       enddo
                   enddo
                enddo
            CLOSE(UNIT=9100+geid)
            isprint = 0
          endif
        enddo
        endif
      end


c-----------------------------------------------------------------------
      subroutine printUarray
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      real   xerange(2,3,lelt)
      common /elementrange/ xerange

      integer isprint, i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2
  
      fmt = '(I4.4)'
      isprint = 0
      geid = 1

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
      if((istep .ge. 1990 .and. istep .le. 2010)
     $      .or.(istep .ge. 2490 .and. istep .le. 2510)
     $      .or.(istep .ge.2990)) then ! .and. isprint .eq. 1) then
c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do geid = 1, nelgt
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            print *, "writing uarray file, istep",istep, 
     >         "id", geid, "location:"
     >     , xerange(1,1,pn), xerange(2,1,pn), xerange(1,2,pn)
     >     , xerange(2,2,pn), xerange(1,3,pn), xerange(2,3,pn)
c    >     , "element", geid, "is in proc", nid, "local", pn

            write(x1, fmt) geid
            write(x2, fmt) istep
          OPEN(UNIT=9999+geid,FILE='uarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
              do i=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=9999+geid, FMT=*) m, n, k, i, u(m,n,k,i,pn)
                       enddo
                   enddo
                enddo
             enddo
            CLOSE(UNIT=9999+geid)
            isprint = 0
          endif
        enddo
        endif
      end


c-----------------------------------------------------------------------
      subroutine printUarray2
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

c     integer stage
      integer isprint, i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3
  
      fmt = '(I4.4)'
      isprint = 0
      geid = 1

c     if(istep .eq. 455 .or. istep .eq. 2550) then ! .and. isprint .eq. 1) then
c     if(mod(istep, 500) .eq. 130 .or.mod(istep,500) .eq. 400) then ! .and. isprint .eq. 1) then
c     if(istep .gt. 500 .and. istep .lt. 630) then ! .and. isprint .eq. 1) then
      if(istep .eq. 502 .or.istep .eq. 501) then ! .and. isprint .eq. 1) then
         do i = 1, nelt
            if(lglel(i) .eq. geid) then !output U array of global element
               isprint = 1              !id = geid
               print *, 'elemt', geid, 'is in proc ', nid, 'local', i
               pn = i
               do j=1, nelt
                  print *, nid,'lglel',j, lglel(j)
               enddo
            endif
         enddo

         if(isprint .eq. 1) then
            print *, "writing uarray file"

            write(x1, fmt) geid
            write(x2, fmt) istep
            write(x3, fmt) stage
          OPEN(UNIT=9999,FILE='uarray'//'.'//'id.'//trim(x1)//'.'
     $        //'step.'//trim(x2)//'stage.'//trim(x3),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
              do i=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=9999, FMT=*) m, n, k, i, u(m,n,k,i,pn)
                       enddo
                   enddo
                enddo
             enddo
            CLOSE(UNIT=9999)
          endif
        endif
      end
