function [KLdiv,KLdiv_rowwise] = KLtransition_dist(A1,A2)
% Compute symmetric KL-divergence between two transition matrices with 
% sufficient statistics A1 and A2
assert(all(size(A1)==size(A2)))

K = size(A1,1); % number of states
KLdiv_rowwise = nan(K,1);
for k = 1:K
    kl12 = dirichlet_kl(A1(k,:),A2(k,:));
    kl21 = dirichlet_kl(A2(k,:),A1(k,:));
    KLdiv_rowwise(k) = (kl12+kl21)/2;
end
KLdiv = sum(KLdiv_rowwise);
%eof
end