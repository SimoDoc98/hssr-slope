function RC = SSA(X,M)
% SSA  Singular Spectrum Analysis of a time series.
%
%   RC = SSA(X,M) returns the M reconstructed components of the signal X
%   (N-by-M matrix), sorted by decreasing eigenvalue of the lag-covariance
%   matrix. M is the embedding dimension (maximum lag).
%
%   Reference: R. Vautard, M. Ghil, Singular spectrum analysis in nonlinear
%   dynamics, with applications to paleoclimatic time series, Physica D 35
%   (1989) 395-424.

N = length(X);

X = X - mean(X);
X = X/std(X,1);

% Toeplitz lag-covariance matrix
covX = xcorr(X,M-1,'unbiased');
C = toeplitz(covX(M:end));

% Time-delayed embedding of X
Y = zeros(N-M+1,M);

for m = 1:M
    Y(:,m) = X((1:N-M+1)+m-1);
end

% Eigenvectors sorted by decreasing eigenvalue
[RHO,LAMBDA] = eig(C);
LAMBDA = diag(LAMBDA);
[~,ind] = sort(LAMBDA,'descend');
RHO = RHO(:,ind);

% Principal components
PC = Y*RHO;

% Reconstructed components (diagonal averaging)
RC = zeros(N,M);

for m = 1:M

    buf = PC(:,m)*RHO(:,m)';
    buf = buf(end:-1:1,:);

    for n = 1:N
        RC(n,m) = mean(diag(buf,-(N-M+1)+n));
    end

end

end
