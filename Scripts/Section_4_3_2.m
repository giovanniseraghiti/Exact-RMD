% ================================================================
% Script supporting the contribution
%     "The ReLU Matrix Decomposition: NP-Hardness, Ill-Posedness, and ReLU-Rank"  
% authors: Giovanni Seraghiti, Arnaud Vandaele, and Nicolas Gillis
% ================================================================
%
% This script computes rank-3 RMD approximation of the identity matrix of
% dimension m, using the Naive-RMD algorithm. Our aim is to show that, 
% even though a rank-3 exact RMD exists, Naive-RMD fails when randomly
% initialized. 
% ================================================================

clear; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'Algorithms_and_utils'));
% ---- Parameters ----
nrep   = 5;          % number of runs
tolSucc = 1e-5;       % Iteration is declares successful if relative error < tolSucc

param = struct();
param.maxit   = 1e5;
param.tol     = 1e-12;
param.time= 30;
param.tol_err = 1e-12;
param.display = 0;    

m = 10; r = 3;
X = eye(m);
normX = norm(X,'fro');

% ---- Run ----
relErr = zeros(nrep,1);
iters  = zeros(nrep,1);
times  = zeros(nrep,1);

for rep = 1:nrep
    rng(rep);                        
    param.W0 = randn(m,r);
    param.H0 = randn(r,m);

    [Theta, err, it, time] = Naive_NMD(X, r, param);

    relErr(rep) = norm(max(0,Theta) - X,'fro') / normX;
    iters(rep)  = it;
    times(rep)  = time(end);

    fprintf('Run %2d: errore relativo = %.3e, iterazioni = %d, tempo = %.2f s\n', ...
        rep, relErr(rep), iters(rep), times(rep));
end

% ---- Statistichs ----
succ = relErr < tolSucc;

fprintf('\n================ RISULTATI ================\n');
fprintf('Errore relativo medio:            %.3e\n', mean(relErr));
fprintf('Errore relativo mediano:          %.3e\n', median(relErr));
fprintf('Successi (errore < %.0e):        %d / %d\n', tolSucc, sum(succ), nrep);
if any(succ)
    fprintf('Tempo medio (run con successo):   %.2f s\n', mean(times(succ)));
    fprintf('Iterazioni medie (con successo):  %.0f\n', mean(iters(succ)));
end