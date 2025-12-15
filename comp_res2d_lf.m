function [res_hjb,res_fp,outs] ...
    = comp_res2d_lf(xmesh,ymesh,tmesh,rho,phi,vis,vis_num,gamma,func_f,func_g)
% compute MFG system residual with Lax-Friedrichs Hamiltonian

dx = xmesh(2,1,1) - xmesh(1,1,1);
dy = ymesh(1,2,1) - ymesh(1,1,1);
dt = tmesh(1,1,2) - tmesh(1,1,1);
func_H = @(px,py) 1/gamma*(sqrt(px.^2+py.^2)).^gamma;
l2norm = @(x) sqrt(sum(x.^2,'all')*dx*dy*dt);

dtphi = op_dt(phi);
lapxphi = op_lapx(phi(:,:,1:end-1));
lapyphi = op_lapy(phi(:,:,1:end-1));
[gradxphi,gradyphi] = op_grad(phi(:,:,1:end-1));
hamphi = func_H(gradxphi,gradyphi);
f = func_f(xmesh,ymesh,tmesh,rho);
f = f(:,:,2:end);
g = func_g(xmesh(:,:,end),ymesh(:,:,end),rho(:,:,end));

res_hjb = -dtphi - (vis+vis_num*dx)*lapxphi ...
                 - (vis+vis_num*dy)*lapyphi + hamphi - f;
res_hjb = cat(3,res_hjb, g - phi(:,:,end) );
outs.res_hjb_mesh = res_hjb;
res_hjb = l2norm(res_hjb(2:end-1,2:end-1,:));

dtrho = op_dt(rho);
lapxrho = op_lapx(rho(:,:,2:end));
lapyrho = op_lapy(rho(:,:,2:end));
[vx,vy] = grad_H(gradxphi,gradyphi);
vx = -vx; vy = -vy;
gradrhov = op_div(rho(:,:,2:end).*vx,rho(:,:,2:end).*vy);

res_fp = dtrho - (vis+vis_num*dx)*lapxrho ...
               - (vis+vis_num*dy)*lapyrho + gradrhov;
outs.res_fp_mesh = res_fp;
res_fp = l2norm(res_fp(2:end-1,2:end-1,:));

%%
    function [gradxH,gradyH] = grad_H(px,py)
        pnorm = sqrt(px.^2+py.^2);
        gradxH = zeros(size(pnorm));
        gradyH = zeros(size(pnorm));
        pnormterm = pnorm.^gamma./pnorm.^2;
        ind = pnorm~=0;
        gradxH(ind) = px(ind).*pnormterm(ind);
        gradyH(ind) = py(ind).*pnormterm(ind);
    end

    function dtu = op_dt(u)
        dtu = (u(:,:,2:end)-u(:,:,1:end-1))/dt;
    end
    
    function dxu = op_gradx(u)
        dxu = cat(1,u(2,:,:)    -u(end-1,:,:),...
                    u(3:end,:,:)-u(1:end-2,:,:),...
                    u(2,:,:)    -u(end-1,:,:))/(2*dx);
    end

    function dyu = op_grady(u)
        dyu = cat(2,u(:,2,:)    -u(:,end-1,:),...
                    u(:,3:end,:)-u(:,1:end-2,:),...
                    u(:,2,:)    -u(:,end-1,:))/(2*dy);
    end

    function [dxu,dyu] = op_grad(u)
        dxu = op_gradx(u);
        dyu = op_grady(u);
    end

    function divm = op_div(mx,my)
        divm = op_gradx(mx) + op_grady(my);
    end

    function lapxu = op_lapx(u)
        lapxu = cat(1,u(end-1,  :,:) - 2*u(1,      :,:) + u(2,    :,:),...
                      u(1:end-2,:,:) - 2*u(2:end-1,:,:) + u(3:end,:,:),...
                      u(end-1,  :,:) - 2*u(end,    :,:) + u(2,    :,:))/(dx^2);
    end

    function lapyu = op_lapy(u)
        lapyu = cat(2,u(:,end-1,  :) - 2*u(:,1,      :) + u(:,2,    :), ...
                      u(:,1:end-2,:) - 2*u(:,2:end-1,:) + u(:,3:end,:),...
                      u(:,end-1,  :) - 2*u(:,end,    :) + u(:,2,    :))/(dy^2);
    end

end