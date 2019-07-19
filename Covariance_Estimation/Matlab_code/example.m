p  = 10000; n = 50; rho = 0.1;

Sigma=0.5*eye(p,p);
for i=1:p
    for j=i+1:p
        Sigma(i,j)=max(0,1-(j-i)/10);
    end
end
Sigma=Sigma+Sigma';

mu=zeros(p,1);
X = mvnrnd(mu,Sigma,n);
Rn = cov(zscore(X));

%%%% generate the weight matrix %%%%
index1 = find( abs(Rn)<= 1.0e-6);
index2 = setdiff(1:p^2,index1);
W = zeros(p,p);
W(index2) = 1./abs(Rn(index2));
W = W-diag(diag(W));

OPTIONS.W = W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = sec(Rn,rho,OPTIONS)

