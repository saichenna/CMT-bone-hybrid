      subroutine nekmxm(a,n1,b,n2,c,n3)
c
c     Compute matrix-matrix product C = A*B
c     for contiguously packed matrices A,B, and C.
c
      real a(n1,n2),b(n2,n3),c(n1,n3)
c
      include 'SIZE'
      include 'OPCTR'
      include 'TOTAL'
c
      integer aligned
      integer K10_mxm
      call mxmf2(a,n1,b,n2,c,n3)

      return
      end
