function KLdiv = KLtransition_dist(A1,A2)
% Compute KL-divergence between to transition matrices with sufficient
% statistics A1 and A2
assert(all(size(A1)==size(A2)))
KLdiv = 0; 
% KL-divergence for transition prob
K = size(A1,1);
for k = 1:K
    KLdiv = KLdiv + dirichlet_kl(A1(k,:),A2(k,:));
end

end