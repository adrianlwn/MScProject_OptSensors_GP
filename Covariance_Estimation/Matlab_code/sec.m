function R = sec(Rn,rho,OPTIONS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                   This code is designed to solve                      %%
%%             min     0.5*<R-Rn,R-Rn>+rho|W\circ R|                     %%
%%             s.t.    R>=epsilon*I(symmetric and positive definite)     %%
%%                     R_ii = b_ii                                       %%
%%       based on the accelerated proximal gradient algorithm in         %%
%%      'Sparse Estimation of High Dimensional correlation matrix'       %%
%%               by Ying Cui, Chenlei Leng and Defeng Sun                %%


%%  Input:  Rn        The empirical correlation (covariance) matrix      %%
%%          rho       The \ell_1 penalty parameter                       %%
%%          OPTIONS   Parameters in the OPTIONS structure                %%
%%
%%  Output: R         The estimated correlation matrix                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
[p,m]=size(Rn);
%------------------------------------------------------------------------%%
%                 Default parameters of the OPTIONS structure            %%
%------------------------------------------------------------------------%%

epsilon          = 1.0e-5;           %%% perturbation for eigenvalue
W                = ones(p,p);        %%% weight matrix for \ell_1 penalty
tol              = 1.0e-3;           %%% tolerance for termination
max_iter         = 1000;             %%% maximum iterations for APG
Z0               = zeros(p,p);       %%% initial value for dual variable
b                = ones(p,1);        %%% correlation matrix
restart          = 50;               %%% iteration numbers before restart the APG algorithm
line_search_apg  = 1;                %%% line_search = 1: do line search
                                     %%%             = 0: no line search

if exist('OPTIONS')
    if isfield(OPTIONS,'epsilon');           epsilon           = OPTIONS.epsilon;            end
    if isfield(OPTIONS,'W');                 W                 = OPTIONS.W;                  end
    if isfield(OPTIONS,'tol');               tol               = OPTIONS.tol;                end
    if isfield(OPTIONS,'max_iter');          max_iter          = OPTIONS.max_iter;           end
    if isfield(OPTIONS,'Z0');                Z0                = OPTIONS.Z0;                 end
    if isfield(OPTIONS,'b');                 b                 = OPTIONS.b;                  end
    if isfield(OPTIONS,'restart');           restart           = OPTIONS.restart;            end
    if isfield(OPTIONS,'line_search_apg');   line_search_apg   = OPTIONS.line_search_apg;    end
end

        
Rn = zscore(Rn);
Rn = (Rn+Rn')/2;  
Z  = Z0;                   %%%   initial dual variable
R  = zeros(p,p);           %%%   initial primal variable
Y  = Z;
t  = 1;
L  = 1;                    %%% Lipschitz constant, which could be used for step length
tau = 0.75;
eta = 0.9;                 %%% back tracking

k = 1;


while (k <= max_iter)
    
    Yold = Y;
    told = t;
    
    X = Z + Rn ;
    
    R = sign(X).*max(abs(X)-rho*W, 0);
    R = R-diag(diag(R))+ diag(b);
    
    if line_search_apg == 1   %%%% do line search
               
       if rem(k,5) == 0   && tau < L      %%%% check the majorization in APG every 5 steps
            if res_gradient > res_old
                tau = min(L,tau/eta);
            end
        end
        Y = projection(Z -(R-epsilon*eye(p))/tau);
        res_gradient =tau* norm(Z-Y,'fro')/(1+norm(Z,'fro'));
    else   %%% no line search
        Y = projection(Z -(R-epsilon*eye(p))/L);
        res_gradient =norm(Z-Y,'fro')/(norm(Z,'fro')+1);
    end
    
    if k >1
        res_old = res_gradient;
    end
    
    
    if res_gradient <= tol
        
        %%%%--------------calibrate the solution-----------------%%%
        R = R -epsilon*eye(p);
        R = R + max(0,-min(eig(R)))*eye(p);
        d = diag(R);
        d = (b-epsilon*ones(p,1))./d;
        d = d.^0.5;
        D = diag(d);
        
        R = sparse(D)*R*sparse(D);
        R = R + epsilon*eye(p);
        break,
    end
 
  
    if rem(k,restart)==0
        t = 1;
        told = t;
    end
    
    t = (1 + sqrt(1+4*t^2))/2;
    ttemp = (told-1)/t;
    Z = Y + ttemp*(Y-Yold);
    
    k=k+1;
end


end


%                      end of apg_sec.m                     %%
%-----------------------------------------------------------%%



%-----------------------------------------------------------%%
%    To project a matrix G onto positive definite cone      %%
function [Proj] = projection(G)

G = (G+G')/2;
[p,m] = size(G);

[P,lambda] = eig(G);         %Eig-decomposition X: X1=P*diag(D)%P^T


P = real(P);
lambda = real(diag(lambda));
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
end

Proj = zeros(p,p);
idx.idp = find(lambda>0);
r=length(idx.idp);
if r>0
    if r < p/2
        i = 1;
        while i <= r
            Proj(i,:) = lambda(i)^0.5* P(:,i)';
            i = i+1;
        end
        Proj = Proj'*Proj;
    elseif r==p
        Proj = G;
    else
        i=r+1;
        while i <= p
            Proj(i,:) = (-lambda(i))^0.5* P(:,i)';
            i = i+1;
        end
        Proj = G + Proj'*Proj; 
    end
    
end

end

%                   end of projection.m                 %%
%-------------------------------------------------------%%

