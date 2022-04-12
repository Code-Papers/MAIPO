program test
    implicit none
    integer :: i
    real :: FPRM (5) = (/1, 2, 3, 4, 5/)

    do i = 5, 2, -1
        call sleep(5)
        print *, i
    enddo


end program test