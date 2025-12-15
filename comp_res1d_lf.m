function [res_hjb,res_fp,outs] ...
    = comp_res1d_lf(xmesh,tmesh,rho,phi,vis,vis_num,gamma,func_f,func_g)
% compute MFG system residual with Lax-Friedrichs Hamiltonian

dx = xmesh(2,1) - xmesh(1,1);
dt = tmesh(1,2) - tmesh(1,1);
func_H = @(p) 1/gamma*abs(p).^gamma;
grad_H = @(p) sign(p).*abs(p).^(gamma-1);
l2norm = @(x) sqrt(sum(x.^2,'all')*dx*dt);

dtphi = op_dt(phi);
lapphi = op_lap(phi(:,1:end-1));
gradphi = op_dx(phi(:,1:end-1));
hamphi = func_H(gradphi);
f = func_f(xmesh,tmesh,rho);
f = f(:,2:end);
g = func_g(xmesh(:,end),rho(:,end));

res_hjb = -dtphi - (vis+vis_num*dx)*lapphi + hamphi - f;
res_hjb = cat(2,res_hjb, g - phi(:,end) );
outs.res_hjb_mesh = res_hjb;
res_hjb = l2norm(res_hjb);

dtrho = op_dt(rho);
laprho = op_lap(rho(:,2:end));
v = -grad_H(gradphi);
gradrhov = op_dx(rho(:,2:end).*v);

res_fp = dtrho - (vis+vis_num*dx)*laprho + gradrhov;
outs.res_fp_mesh = res_fp;
res_fp = l2norm(res_fp(2:end-1,:));

%%
    function dtu = op_dt(u)
        dtu = (u(:,2:end)-u(:,1:end-1))/dt;
    end
    
    function dxu = op_dx(u)
        dxu = cat(1,u(2,:)    -u(end-1,:),...
                    u(3:end,:)-u(1:end-2,:),...
                    u(2,:)    -u(end-1,:))/(2*dx);
    end

    function lapxu = op_lap(u)
        lapxu = cat(1,u(end-1,:)   - 2*u(1,:)       + u(2,:),...
                      u(1:end-2,:) - 2*u(2:end-1,:) + u(3:end,:),...
                      u(end-1,:)   - 2*u(end,:)     + u(2,:))/(dx^2);
    end

end