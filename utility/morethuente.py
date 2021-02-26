import numpy as np
import math

#function [x,f,g,stp,info,nfev] ...
def cvsrch(fcn, n, x, f, g, s, stp, ftol, gtol, xtol, stpmin, stpmax, maxfev):
    p5 = .5
    p66 = .66
    xtrapf = 4
    info = 0
    infoc = 1

    # Compute the initial gradient in the search direction
    # and check that s is a descent direction.
    dginit = np.dot(g, s)
    if (dginit >= 0.0):
        return x, f, g, 0, 0, 0

    # Initialize local variables.
    brackt = 0
    stage1 = 1
    nfev = 0
    finit = f
    dgtest = ftol*dginit
    width = stpmax - stpmin
    width1 = 2*width
    wa = x

    # The variables stx, fx, dgx contain the values of the step, 
    # function, and directional derivative at the best step.
    # The variables sty, fy, dgy contain the value of the step,
    # function, and derivative at the other endpoint of
    # the interval of uncertainty.
    # The variables stp, f, dg contain the values of the step,
    # function, and derivative at the current step.
    stx = 0.0
    fx = finit
    dgx = dginit
    sty = 0.0
    fy = finit
    dgy = dginit

    # Start of iteration.
    while (1):
        # Set the minimum and maximum steps to correspond
        # to the present interval of uncertainty.
        if (brackt):
            stmin = min(stx,sty)
            stmax = max(stx,sty)
        else:
            stmin = stx
            stmax = stp + xtrapf*(stp - stx)
        
        # Force the step to be within the bounds stpmax and stpmin.
        stp = max(stp,stpmin)
        stp = min(stp,stpmax)
        
        # If an unusual termination is to occur then let 
        # stp be the lowest point obtained so far.
        if ((brackt and (stp <= stmin or stp >= stmax)) \
            or nfev >= maxfev-1 or infoc == 0 or (brackt and stmax-stmin <= xtol*stmax)):
            stp = stx

        # Evaluate the function and gradient at stp
        # and compute the directional derivative.
        x = wa + stp * s
        f, g = fcn(x)
        nfev = nfev + 1
        dg = np.dot(g, s)
        ftest1 = finit + stp*dgtest

        # Test for convergence.
        if ((brackt and (stp <= stmin or stp >= stmax)) or infoc == 0):
            info = 6
        if (stp == stpmax and f <= ftest1 and dg <= dgtest):
            info = 5
        if (stp == stpmin and (f > ftest1 or dg >= dgtest)):
            info = 4
        if (nfev >= maxfev):
            info = 3
        if (brackt and stmax-stmin <= xtol*stmax):
            info = 2
        if (f <= ftest1 and abs(dg) <= gtol*(-dginit)):
            info = 1

        # Check for termination.
        if (info != 0):
            return x, f, g, stp, info, nfev

        # In the first stage we seek a step for which the modified
        # function has a nonpositive value and nonnegative derivative.
        if (stage1 and f <= ftest1 and dg >= min(ftol,gtol)*dginit):
            stage1 = 0

        # A modified function is used to predict the step only if
        # we have not obtained a step for which the modified
        # function has a nonpositive function value and nonnegative 
        # derivative, and if a lower function value has been  
        # obtained but the decrease is not sufficient.
        if (stage1 and f <= fx and f > ftest1):
            # Define the modified function and derivative values.
            fm = f - stp*dgtest
            fxm = fx - stx*dgtest
            fym = fy - sty*dgtest
            dgm = dg - dgtest
            dgxm = dgx - dgtest
            dgym = dgy - dgtest

            # Call cstep to update the interval of uncertainty 
            # and to compute the new step.
            stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, infoc = \
                cstep(stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm, brackt,stmin,stmax)

            # Reset the function and gradient values for f.
            fx = fxm + stx*dgtest
            fy = fym + sty*dgtest
            dgx = dgxm + dgtest
            dgy = dgym + dgtest
        else:
            # Call cstep to update the interval of uncertainty 
            # and to compute the new step.
            stx,fx,dgx,sty,fy,dgy,stp,f,dg,brackt,infoc = \
                cstep(stx,fx,dgx,sty,fy,dgy,stp,f,dg, brackt,stmin,stmax)

        # Force a sufficient decrease in the size of the
        # interval of uncertainty.
        if (brackt):
            if (abs(sty-stx) >= p66*width1):
                stp = stx + p5*(sty - stx)
            width1 = width
            width = abs(sty-stx)




# function  [stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,info] ...
def cstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    p66 = 0.66
    info = 0

    # Determine if the derivatives have opposite sign.
    sgnd = dp*(dx/abs(dx))

    # First case. A higher function value.
    # The minimum is bracketed. If the cubic step is closer
    # to stx than the quadratic step, the cubic step is taken,
    # else the average of the cubic and quadratic steps is taken.
    if (fp > fx):
        info = 1
        bound = 1
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = np.linalg.norm([theta,dx,dp],np.inf)
        gamma = s*math.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if (stp < stx):
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2)*(stp - stx)
        if (abs(stpc-stx) < abs(stpq-stx)):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc)/2
        brackt = 1

    # Second case. A lower function value and derivatives of
    # opposite sign. The minimum is bracketed. If the cubic
    # step is closer to stx than the quadratic (secant) step, 
    # the cubic step is taken, else the quadratic step is taken.
    elif (sgnd < 0.0):
        info = 2
        bound = 0
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = np.linalg.norm([theta,dx,dp],np.inf)
        gamma = s*math.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if (stp > stx):
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p/q
        stpc = stp + r*(stx - stp)
        stpq = stp + (dp/(dp-dx))*(stx - stp)
        if (abs(stpc-stp) > abs(stpq-stp)):
            stpf = stpc
        else:
            stpf = stpq
        brackt = 1

    # Third case. A lower function value, derivatives of the
    # same sign, and the magnitude of the derivative decreases.
    # The cubic step is only used if the cubic tends to infinity 
    # in the direction of the step or if the minimum of the cubic
    # is beyond stp. Otherwise the cubic step is defined to be 
    # either stpmin or stpmax. The quadratic (secant) step is also 
    # computed and if the minimum is bracketed then the the step 
    # closest to stx is taken, else the step farthest away is taken.
    elif (abs(dp) < abs(dx)):
        info = 3
        bound = 1
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = np.linalg.norm([theta,dx,dp],np.inf)

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s*math.sqrt(max(0.,(theta/s)**2 - (dx/s)*(dp/s)))
        if (stp > stx):
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if (r < 0.0 and gamma != 0.0):
            stpc = stp + r*(stx - stp)
        elif (stp > stx):
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp/(dp-dx))*(stx - stp)
        if (brackt):
            if (abs(stp-stpc) < abs(stp-stpq)):
                stpf = stpc
            else:
                stpf = stpq
        else:
            if (abs(stp-stpc) > abs(stp-stpq)):
               stpf = stpc
            else:
               stpf = stpq

    # Fourth case. A lower function value, derivatives of the
    # same sign, and the magnitude of the derivative does
    # not decrease. If the minimum is not bracketed, the step
    # is either stpmin or stpmax, else the cubic step is taken.
    else:
        info = 4
        bound = 0
        if (brackt):
            theta = 3*(fp - fy)/(sty - stp) + dy + dp
            s = np.linalg.norm([theta,dy,dp],np.inf)
            gamma = s*math.sqrt((theta/s)**2 - (dy/s)*(dp/s))
            if (stp > sty):
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpc = stp + r*(sty - stp)
            stpf = stpc
        elif (stp > stx):
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval of uncertainty. This update does not
    # depend on the new step or the case analysis above.
    if (fp > fx):
        sty = stp
        fy = fp
        dy = dp
    else:
        if (sgnd < 0.0):
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step and safeguard it.
    stpf = min(stpmax,stpf)
    stpf = max(stpmin,stpf)
    stp = stpf
    if (brackt and bound):
        if (sty > stx):
            stp = min(stx+p66*(sty-stx),stp)
        else:
            stp = max(stx+p66*(sty-stx),stp)

    return stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,info
