function [KLdiv] = KLpi_dist(A1,A2)
% Compute symmetric KL-divergence between two state distribution vectors
% (pi from hmm object) with sufficient statistics A1 and A2
assert(all(size(A1)==size(A2)))
if size(A1,2)==1, A1 = A1';end % vectors should be row-vectors
if size(A2,2)==1, A2 = A2';end

% Calc Kldist
kl12 = dirichlet_kl(A1,A2);
kl21 = dirichlet_kl(A2,A1);
KLdiv= (kl12+kl21)/2;
%eof
end