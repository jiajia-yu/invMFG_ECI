function q = mg1d_restrc_sp(q)

kernel = [1, 2, 1]/4;

q = [q(end-1); q; q(2)];

q1 = conv(q,kernel,'valid');
q = q1(1:2:end);
