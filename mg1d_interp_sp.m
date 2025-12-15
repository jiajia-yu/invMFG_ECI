function q = mg1d_interp_sp(x,q,xq)

q = interp1(x,q,xq,'spline');

