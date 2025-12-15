function rho = fp2d_lf(rho0,vx,vy,vis,opts)
% solving 1D Fokker-Planck equation 
% using implicit Lax-Friedrichs scheme
% partial_t rho - vis lap(rho) + div(rho*v) = 0
% rho(x,0) = rho0(x)
% implicit numerical viscosity
persistent siz_old dx_old dy_old vis_old vis_num_old cst_mat
persistent ind_xl_c   ind_xl_xp  ind_yl_c   ind_yl_yp ...
           ind_xin_xm ind_xin_c  ind_xin_xp ...
           ind_yin_ym ind_yin_c  ind_yin_yp ...
           ind_xr_xm  ind_xr_c   ind_yr_ym  ind_yr_c
persistent ind_xm ind_ym ind_c ind_xp ind_yp

if isfield(opts,'vis_num') vis_num = opts.vis_num; else vis_num = 1; end
if vis < 1e-6 vis_num = 1; end
if isfield(opts,'xrange') xlend = opts.xrange(1); xrend = opts.xrange(2);
                     else xlend = 0; xrend = 1; end
if isfield(opts,'yrange') ylend = opts.yrange(1); yrend = opts.yrange(2);
                     else ylend = 0; yrend = 1; end
if isfield(opts,'tend') tend = opts.tend; else tend = 1; end
nxp = size(vx,1); nx = nxp-1; nxm = nx-1; dx = (xrend-xlend)/nx;
nyp = size(vx,2); ny = nyp-1; nym = ny-1; dy = (yrend-ylend)/ny;
ntp = size(vx,3); nt = ntp-1; dt = tend/nt;
dtodxsq = dt/(dx)^2;
dtodysq = dt/(dy)^2;
dtodx = dt/dx;
dtody = dt/dy;
xpar_vis = vis*dtodxsq+0.5*vis_num*dtodx;
ypar_vis = vis*dtodysq+0.5*vis_num*dtody;

precompute=0;

if isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    siz = nx * ny;
else
    siz = nxp * nyp;
end
if isempty(cst_mat)
    precompute = 1;
    siz_old = siz;
    dx_old = dx;
    dy_old = dy;
    vis_old = vis;
    vis_num_old = vis_num;
elseif any( abs([siz_old,dx_old,dy_old,vis_old,vis_num_old]...
               -[siz,    dx,    dy,    vis,    vis_num])>0 )
    precompute = 1;
    siz_old = siz;
    dx_old = dx;
    dy_old = dy;
    vis_old = vis;
    vis_num_old = vis_num;
end
if precompute
    if isfield(opts,'bd') && strcmp(opts.bd,'periodic')
        [cst_mat, ind_c, ind_xm, ind_xp, ind_ym, ind_yp] = gen_mat_periodic;
    else
        [cst_mat, ...          
         ind_xl_c,   ind_xl_xp, ind_yl_c,  ind_yl_yp, ...
         ind_xin_xm, ind_xin_c, ind_xin_xp, ...
         ind_yin_ym, ind_yin_c, ind_yin_yp, ...
         ind_xr_xm,  ind_xr_c,  ind_yr_ym, ind_yr_c] = gen_mat_neumann;
    end
end

%% forward in time
rho = zeros(nxp,nyp,ntp);
rho(:,:,1) = rho0;

if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    if isfield(opts,'verbose') && opts.verbose disp('dirichlet bd TBA'); end
    % rho(1,:,:) = opts.rhoxl; rho(end,:,:) = opts.rhoxr;
    % rho(:,1,:) = opts.rhoyl; rho(:,end,:) = opts.rhoyl;
    % cstmat = spdiags(ones(nxm,1)*[-xpar_vis,1+2*xpar_vis,-xpar_vis],...
    %                  [-1,0,1],nxm,nxm);
    % 
    % for idt = 1:nt
    %     rhot = rho(:,idt);
    %     vt = vx(:,idt);
    % 
    %     lhs = cstmat... 
    %          + spdiags([-(0.5*dtodx*vt(2:end-1)),...
    %                      (0.5*dtodx*vt(2:end-1))],[-1,1],nxm,nxm);
    % 
    %     rhs = rhot(2:end-1);
    %     rhs(1) = rhs(1) ...
    %             + (0.5*dtodx*vt(1) + vis*dtodxsq)*rho(1,idt+1);
    %     rhs(end) = rhs(end) ...
    %             + (-0.5*dtodx*vt(end) + vis*dtodxsq)*rho(end,idt+1);
    % 
    %     rhotp1 = lhs\rhs;    
    %     rho(2:end-1,idt+1) = rhotp1;
    % end
    
    
elseif isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    if isfield(opts,'verbose') && opts.verbose disp('periodic'); end

    for idt = 1:nt
        rhot = rho(2:end,2:end,idt);
        vxt = 0.5*dtodx*vx(2:end,2:end,idt);
        vyt = 0.5*dtody*vy(2:end,2:end,idt);
        
        lhs = cst_mat; 
        lhs = lhs - sparse(ind_c,ind_xm,vxt(ind_xm), siz,siz);
        lhs = lhs - sparse(ind_c,ind_ym,vyt(ind_ym), siz,siz);
        lhs = lhs + sparse(ind_c,ind_xp,vxt(ind_xp), siz,siz);
        lhs = lhs + sparse(ind_c,ind_yp,vyt(ind_yp), siz,siz);       
        

        rhs = reshape(rhot,[],1);
        
        rhotp1 = reshape(lhs\rhs,nx,ny);    
        rho(:,:,idt+1) = [rhotp1(end,end),rhotp1(end,:);
                          rhotp1(:,end),  rhotp1];
    end
    
    
% default boundary condition: zero-flux
else %if isfield(opts,'bd') && strcmp(opts.bd,'zero_flux')
    if isfield(opts,'verbose') && opts.verbose disp('zero flux'); end

    for idt = 1:nt
        rhot = rho(:,:,idt);
        vxt = 0.5*dtodx*vx(:,:,idt);
        vyt = 0.5*dtody*vy(:,:,idt);
        
        lhs = cst_mat;
        % x left boundary
        lhs = lhs + sparse(ind_xl_c, ind_xl_c,  vxt(ind_xl_c),  siz,siz);
        lhs = lhs + sparse(ind_xl_c, ind_xl_xp, vxt(ind_xl_xp), siz,siz);
        % y left boundary
        lhs = lhs + sparse(ind_yl_c, ind_yl_c,  vyt(ind_yl_c),  siz,siz);
        lhs = lhs + sparse(ind_yl_c, ind_yl_yp, vyt(ind_yl_yp), siz,siz);
        % x interior
        lhs = lhs - sparse(ind_xin_c,ind_xin_xm,vxt(ind_xin_xm),siz,siz);
        lhs = lhs + sparse(ind_xin_c,ind_xin_xp,vxt(ind_xin_xp),siz,siz);
        % y interior
        lhs = lhs - sparse(ind_yin_c,ind_yin_ym,vyt(ind_yin_ym),siz,siz);
        lhs = lhs + sparse(ind_yin_c,ind_yin_yp,vyt(ind_yin_yp),siz,siz);
        % x right boundary
        lhs = lhs - sparse(ind_xr_c, ind_xr_xm, vxt(ind_xr_xm), siz,siz);
        lhs = lhs - sparse(ind_xr_c, ind_xr_c,  vxt(ind_xr_c),  siz,siz);
        % y right boundary
        lhs = lhs - sparse(ind_yr_c, ind_yr_ym, vyt(ind_yr_ym), siz,siz);
        lhs = lhs - sparse(ind_yr_c, ind_yr_c,  vyt(ind_yr_c),  siz,siz);
        
        rhs = reshape(rhot,[],1);
        
        rhotp1 = reshape(lhs\rhs,nxp,nyp);    
        rho(:,:,idt+1) = rhotp1;
    end
end

%% -----------------------------------------------------------
    function [cst_mat, ...          
              ind_c, ind_xm, ind_xp, ind_ym, ind_yp] = gen_mat_periodic
        cst_mat = speye(siz,siz);
        
        sub_xm = repmat([nx,1:nx-1]', 1,ny);
        sub_xc = repmat((1:nx)',1,ny);
        sub_xp = repmat([2:nx,1]',    1,ny);
        
        sub_ym = repmat([ny,1:ny-1], nx,1);
        sub_yc = repmat((1:ny), nx,1);
        sub_yp = repmat([2:ny,1],    nx,1);
        
        ind_c  = sub2ind([nx,ny],sub_xc(:),sub_yc(:));
        ind_xm = sub2ind([nx,ny], sub_xm(:), sub_yc(:));
        ind_xp = sub2ind([nx,ny], sub_xp(:), sub_yc(:));
        ind_ym = sub2ind([nx,ny], sub_xc(:), sub_ym(:));
        ind_yp = sub2ind([nx,ny], sub_xc(:), sub_yp(:));
        
        cst_mat = cst_mat + sparse(ind_c, ind_xm, ...
                                   -xpar_vis*ones(siz,1), siz,siz);
        cst_mat = cst_mat + sparse(ind_c, ind_ym, ...
                                   -ypar_vis*ones(siz,1), siz,siz);
        cst_mat = cst_mat + sparse(ind_c, ind_c, ...
                                   2*(xpar_vis+ypar_vis)*ones(siz,1), siz,siz);
        cst_mat = cst_mat + sparse(ind_c, ind_xp, ...
                                   -xpar_vis*ones(siz,1), siz,siz);
        cst_mat = cst_mat + sparse(ind_c, ind_yp, ...
                                   -ypar_vis*ones(siz,1), siz,siz);
        
    end

    function [cst_mat, ...          
              ind_xl_c,   ind_xl_xp, ind_yl_c,  ind_yl_yp, ...
              ind_xin_xm, ind_xin_c, ind_xin_xp, ...
              ind_yin_ym, ind_yin_c, ind_yin_yp, ...
              ind_xr_xm,  ind_xr_c,  ind_yr_ym, ind_yr_c] = gen_mat_neumann

        cst_mat = speye(siz,siz);

        % x left boundary
        sub_xl_x = repmat(1,       1, nyp);
        sub_xl_y = repmat((1:nyp), 1, 1);
        ind_xl_c  = sub2ind([nxp,nyp], sub_xl_x(:),   sub_xl_y(:));
        ind_xl_xp = sub2ind([nxp,nyp], sub_xl_x(:)+1, sub_xl_y(:));
        cst_mat = cst_mat + sparse(ind_xl_c, ind_xl_c, ...
                                    xpar_vis*ones(length(ind_xl_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_xl_c, ind_xl_xp, ...
                                   -xpar_vis*ones(length(ind_xl_c),1), siz,siz);
        % y left boundary
        sub_yl_x = repmat((1:nxp)',1,   1);
        sub_yl_y = repmat(1,       nxp, 1);
        ind_yl_c  = sub2ind([nxp,nyp], sub_yl_x(:), sub_yl_y(:));
        ind_yl_yp = sub2ind([nxp,nyp], sub_yl_x(:), sub_yl_y(:)+1);
        cst_mat = cst_mat + sparse(ind_yl_c, ind_yl_c, ...
                                    ypar_vis*ones(length(ind_yl_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_yl_c, ind_yl_yp, ...
                                   -ypar_vis*ones(length(ind_yl_c),1), siz,siz);
        % x interior
        sub_xin_x = repmat((2:nx)', 1,  nyp);
        sub_xin_y = repmat((1:nyp), nxm,1);
        ind_xin_xm = sub2ind([nxp,nyp], sub_xin_x(:)-1, sub_xin_y(:));
        ind_xin_c  = sub2ind([nxp,nyp], sub_xin_x(:),   sub_xin_y(:));
        ind_xin_xp = sub2ind([nxp,nyp], sub_xin_x(:)+1, sub_xin_y(:));
        cst_mat = cst_mat + sparse(ind_xin_c, ind_xin_xm, ...
                                   -xpar_vis*ones(length(ind_xin_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_xin_c, ind_xin_c, ...
                                    2*xpar_vis*ones(length(ind_xin_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_xin_c, ind_xin_xp, ...
                                   -xpar_vis*ones(length(ind_xin_c),1), siz,siz);
        % y interior
        sub_yin_x = repmat((1:nxp)', 1,  nym);
        sub_yin_y = repmat((2:ny),   nxp,1);
        ind_yin_ym = sub2ind([nxp,nyp], sub_yin_x(:), sub_yin_y(:)-1);
        ind_yin_c  = sub2ind([nxp,nyp], sub_yin_x(:), sub_yin_y(:));
        ind_yin_yp = sub2ind([nxp,nyp], sub_yin_x(:), sub_yin_y(:)+1);
        cst_mat = cst_mat + sparse(ind_yin_c, ind_yin_ym, ...
                                   -ypar_vis*ones(length(ind_yin_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_yin_c, ind_yin_c, ...
                                    2*ypar_vis*ones(length(ind_yin_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_yin_c, ind_yin_yp, ...
                                   -ypar_vis*ones(length(ind_yin_c),1), siz,siz);
        % x right boundary
        sub_xr_x = repmat(nxp,     1, nyp);
        sub_xr_y = repmat((1:nyp), 1, 1);
        ind_xr_xm = sub2ind([nxp,nyp], sub_xr_x(:)-1, sub_xr_y(:));
        ind_xr_c  = sub2ind([nxp,nyp], sub_xr_x(:),   sub_xr_y(:));
        cst_mat = cst_mat + sparse(ind_xr_c, ind_xr_xm, ...
                                   -xpar_vis*ones(length(ind_xr_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_xr_c, ind_xr_c, ...
                                    xpar_vis*ones(length(ind_xr_c),1), siz,siz);
        % y right boundary
        sub_yr_x = repmat((1:nxp)', 1, 1);
        sub_yr_y = repmat(nyp,      nxp, 1);
        ind_yr_ym = sub2ind([nxp,nyp], sub_yr_x(:), sub_yr_y(:)-1);
        ind_yr_c  = sub2ind([nxp,nyp], sub_yr_x(:), sub_yr_y(:));
        cst_mat = cst_mat + sparse(ind_yr_c, ind_yr_ym, ...
                                   -ypar_vis*ones(length(ind_yr_c),1), siz,siz);
        cst_mat = cst_mat + sparse(ind_yr_c, ind_yr_c, ...
                                    ypar_vis*ones(length(ind_yr_c),1), siz,siz);

    end

end

