% ================================================================
% Script supporting the contribution
%     "The ReLU Matrix Decomposition: NP-Hardness, Ill-Posedness, and ReLU-Rank"  
% authors: Giovanni Seraghiti, Arnaud Vandaele, and Nicolas Gillis
% ================================================================
%
% This Script numerically verifies proposition 2.1 in the reference above,
% showing the ill-posedness of the Frobenius ReLU Matrix Decomposition(Fro-RMD).
%
% Specifically, we run the Naive-RMD algorithm on a 3x3 matrix with rank=2,
% and we show that the algorithm seems to stagnates while the infinity norm
% of the approximation monotonically increasing. This experiments given the
% intuition that RMD might be ill-posed in the sense that the minimum might
% not be atained.
% ================================================================


close all
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'Algorithms_and_utils'));

rng(3)

X=[1,1,1;
    1,1,2;
    1,2,0];

W0=randn(3,2); H0=randn(2,3);
param.Theta0=W0*H0;

param.maxit=5000; 
r=2;
[Theta,err,normInf]=Naive_NMD_mod_illposed(X,r,param);

% Colors
colorErr = [0 0.4470 0.7410];      
colorNorm = [0.8500 0.3250 0.0980]; 

figure;
hold on;
grid on;

% left axes: error
yyaxis left
semilogy(err,'-','LineWidth',2,'Color',colorErr);
ylabel('Error','Color',colorErr)
ax = gca;
ax.YColor = colorErr;
ylim([err(end)-0.1 err(1)+0.1])

% right axes: infinity norm
yyaxis right
plot(normInf,'-','LineWidth',2,'Color',colorNorm);
ylabel('$\|\Theta^k\|_\infty$','Interpreter','latex','Color',colorNorm,'FontSize',20,'FontName','Times New Roman')
ax.YColor = colorNorm;

xlabel('Iteration','FontSize',20,'FontName','Times New Roman')
title('Error and infinity norm','FontSize',20,'FontName','Times New Roman')

legend('Rel Err','$\|\Theta^k\|_\infty$','Interpreter','latex','Location','best','FontSize',20,'FontName','Times New Roman')
set(gca,'FontSize',20,'FontName','Times New Roman')
