function phi = hjb1d_revt_lf(phiend,vis,gamma,f,opts)
% solving 1D HJB equation 
% using implicit Lax-Friedrichs method
% -partial_t phi -vis lap(phi) + 1/gamma|grad(phi)|^gamma = f(x,t)
% phi(x,T) = phiend(x)

if isfield(opts,'xrange') lend = opts.xrange(1); rend = opts.xrange(2);
                     else lend = 0; rend = 1; end
if isfield(opts,'tend') tend = opts.tend; else tend = 1; end
if isfield(opts,'vis_num') vis_num = opts.vis_num; else vis_num = 1; end
if vis < 1e-6 vis_num = 1; end
if isfield(opts,'ntit') ntit = opts.ntit; else ntit = 10; end
if isfield(opts,'nttol') nttol = opts.nttol; else nttol = 1e-4; end

nxp = size(f,1); nx = nxp-1; nxm = nx-1; dx = (rend-lend)/nx;
ntp = size(f,2); nt = ntp-1; dt = tend/nt;
dtodxsq = dt/(dx)^2;
dtodx = dt/dx;
par_vis = vis*dtodxsq+0.5*vis_num*dtodx;
funcH = @(p) 1/gamma*abs(p).^gamma;
gradH = @(p) sign(p).*abs(p).^(gamma-1);

% backward in time
phi = zeros(nxp,ntp);
phi(:,end) = phiend;

if isfield(opts,'bd') && strcmp(opts.bd,'dirichlet')
    if isfield(opts,'verbose') && opts.verbose disp('dirichlet'); end
    cstmat = spdiags(ones(nxm,1)*[-par_vis,1+2*par_vis,-par_vis],[-1,0,1],nxm,nxm);
    phi(1,:) = opts.phil;
    phi(end,:) = opts.phir;
    for idt = nt:-1:1
        phitp = phi(:,idt+1);
        ftp = f(:,idt+1);
        phit = phitp; phit(1) = phi(1,idt); phit(end) = phi(end,idt);
        phit = newton_dc(phit,phitp,ftp); 
        phi(:,idt) = phit;
    end

elseif isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    if isfield(opts,'verbose') && opts.verbose disp('periodic'); end
    cstmat = spdiags(ones(nx,1)*[-par_vis,1+2*par_vis,-par_vis],[-1,0,1],nx,nx);
    cstmat(1,end) = -par_vis; cstmat(end,1) = -par_vis;
    for idt = nt:-1:1
        phitp = phi(:,idt+1);
        ftp = f(:,idt+1);
        phit = phitp(2:end); 
        phit = newton_pd(phit,phitp(2:end),ftp(2:end)); 
        phi(:,idt) = [phit(end);phit];
    end
    
% default boundary condition: homogeneous Neumann boundary condition
else %if isfield(opts,'bd') && strcmp(opts.bd,'periodic')
    if isfield(opts,'verbose') && opts.verbose disp('neumann'); end
    cstmat = spdiags(ones(nxp,1)*[-par_vis,1+2*par_vis,-par_vis],...
                     [-1,0,1],nxp,nxp);
    cstmat(1,1) = 1+par_vis; cstmat(end,end) = 1+par_vis;    
    for idt = nt:-1:1
        phitp = phi(:,idt+1);
        ftp = f(:,idt+1);
        phit = phitp; 
        phit = newton_nm(phit,phitp,ftp); 
        phi(:,idt) = phit;
    end
end
%-----------------------------------------------------------
    function phit = newton_dc(phit,phitp,ft)
        for int = 1:ntit
            gradphi = (phit(3:end)-phit(1:end-2))/(2*dx);
            gradH_gradphi = gradH(gradphi);
            h = cstmat*phit(2:end-1) + dt*funcH(gradphi) ...
                - phitp(2:end-1) - dt*ft(2:end-1);
            h(1) = h(1) - par_vis*phit(1);
            h(end) = h(end) - par_vis*phit(end);
            if max(abs(h)) < nttol
                break
            end
            Jh = cstmat - spdiags(0.5*dtodx*gradH_gradphi,1, nxm,nxm)'...
                        + spdiags(0.5*dtodx*gradH_gradphi,-1,nxm,nxm)';
            phit(2:end-1) = phit(2:end-1) - Jh\h;
        end
    end

    function phit = newton_pd(phit,phitp,ft)
        for int = 1:ntit
            gradphi = [phit(2)-phit(end);
                       phit(3:end)-phit(1:end-2);
                       phit(1)-phit(end-1)]/(2*dx);
            gradH_gradphi = gradH(gradphi);
            h = cstmat*phit + dt*funcH(gradphi) - phitp - dt*ft;
            if max(abs(h)) < nttol
                break
            end
            Jh = cstmat - spdiags(0.5*dtodx*gradH_gradphi,1, nx,nx)'...
                        + spdiags(0.5*dtodx*gradH_gradphi,-1,nx,nx)';
            Jh(1,end) = Jh(1,end) - 0.5*dtodx*gradH_gradphi(1);
            Jh(end,1) = Jh(end,1) + 0.5*dtodx*gradH_gradphi(end);
            phit = phit - Jh\h;
        end
    end

    function phit = newton_nm(phit,phitp,ft)
        for int = 1:ntit
            gradphi = [phit(2)-phit(1);
                       phit(3:end)-phit(1:end-2);
                       phit(end)-phit(end-1)]/(2*dx);
            gradH_gradphi = gradH(gradphi);
            h = cstmat*phit + dt*funcH(gradphi) - phitp - dt*ft;
            if max(abs(h)) < nttol
                break
            end
            Jh = cstmat - spdiags(0.5*dtodx*gradH_gradphi,1, nxp,nxp)'...
                        + spdiags(0.5*dtodx*gradH_gradphi,-1,nxp,nxp)';
            Jh(1,1) = Jh(1,1) - 0.5*dtodx*gradH_gradphi(1);
            Jh(end,end) = Jh(end,end) + 0.5*dtodx*gradH_gradphi(end);
            phit = phit - Jh\h;
        end        
    end

end
