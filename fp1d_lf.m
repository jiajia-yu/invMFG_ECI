function rho = fp1d_lf(rho0,v,vis,opts)
% solving 1D Fokker-Planck equation 
% using implicit Lax-Friedrichs scheme
% partial_t rho - vis lap(rho) + div(rho*v) = 0
% rho(x,0) = rho0(x)
% implicit numerical viscosity

if isfield(opts,'vis_num') vis_num = opts.vis_num; else vis_num = 1; end
if vis < 1e-6 vis_num = 1; end
if isfield(opts,'xrange') lend = opts.xrange(1); rend = opts.xrange(2);
                     else lend = 0; rend = 1; end
if isfield(opts,'tend') tend = opts.tend; else tend = 1; end
nxp = size(v,1); nx = nxp-1; nxm = nx-1; dx = (rend-lend)/nx;
ntp = size(v,2); nt = ntp-1; dt = tend/nt;
dtodxsq = dt/(dx)^2;
dtodx = dt/dx;
par_vis = vis*dtodxsq+0.5*vis_num*dtodx;

rho = zeros(nxp,ntp);
rho(:,1) = rho0;

% forward in time
if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    if isfield(opts,'verbose') && opts.verbose disp('dirichlet'); end
    rho(1,:) = opts.rhol;
    rho(end,:) = opts.rhor;
    cstmat = spdiags(ones(nxm,1)*[-par_vis,1+2*par_vis,-par_vis],...
                     [-1,0,1],nxm,nxm);
    
    for idt = 1:nt
        rhot = rho(:,idt);
        vt = v(:,idt);
        
        lhs = cstmat... 
             + spdiags([-(0.5*dtodx*vt(2:end-1)),...
                         (0.5*dtodx*vt(2:end-1))],[-1,1],nxm,nxm);

        rhs = rhot(2:end-1);
        rhs(1) = rhs(1) ...
                + (0.5*dtodx*vt(1) + vis*dtodxsq)*rho(1,idt+1);
        rhs(end) = rhs(end) ...
                + (-0.5*dtodx*vt(end) + vis*dtodxsq)*rho(end,idt+1);
        
        rhotp1 = lhs\rhs;    
        rho(2:end-1,idt+1) = rhotp1;
    end
    
    
elseif isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    if isfield(opts,'verbose') && opts.verbose disp('periodic'); end
    cstmat = spdiags(ones(nx,1)*[-par_vis,1+2*par_vis,-par_vis],...
                     [-1,0,1],nx,nx);
    cstmat(1,end) = -par_vis; cstmat(end,1) = -par_vis;
    
    for idt = 1:nt
        rhot = rho(:,idt);
        vt = v(:,idt);
        
        lhs = cstmat...
             + spdiags([-(0.5*dtodx*vt(2:end)),...
                         (0.5*dtodx*vt(2:end))],[-1,1],nx,nx);
        lhs(1,end) = cstmat(1,end) -(0.5*dtodx*vt(end)); 
        lhs(end,1) = cstmat(end,1) +(0.5*dtodx*vt(2));

        rhs = rhot(2:end);
        
        rhotp1 = lhs\rhs;    
        rho(:,idt+1) = [rhotp1(end);rhotp1];
    end
    
    
% default boundary condition: zero-flux
else %if isfield(opts,'bd') && strcmp(opts.bd,'zero_flux')
    if isfield(opts,'verbose') && opts.verbose disp('zero flux'); end
    cstmat = spdiags(ones(nxp,1)*[-par_vis,1+2*par_vis,-par_vis],...
                     [-1,0,1],nxp,nxp);
    cstmat(1,1) = 1+par_vis; cstmat(end,end) = 1+par_vis;    
    for idt = 1:nt
        rhot = rho(:,idt);
        vt = v(:,idt);
        
        lhs = cstmat ...
             + spdiags([-(0.5*vt*dtodx),...
                        (0.5*vt*dtodx)],[-1,1],nxp,nxp);
        lhs(1,1) = cstmat(1,1) + 0.5*dtodx*vt(1); 
        lhs(end,end) = cstmat(end,end) - 0.5*vt(end)*dtodx;
        
        rhs = rhot;
        
        rhotp1 = lhs\rhs;    
        rho(:,idt+1) = rhotp1;
    end
end


end

