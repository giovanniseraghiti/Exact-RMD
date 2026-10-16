% ================================================================
% Script supporting the contribution
%     "The ReLU Matrix Decomposition: NP-Hardness, Ill-Posedness, and ReLU-Rank"  
% authors: Giovanni Seraghiti, Arnaud Vandaele, and Nicolas Gillis
% ================================================================
% 
%  model built with YALMIP
%
%  FIND W AND H SUCH THAT X = MAX(0,WH) WITH X THE IDENTITY MATRIX
%
% Formulate the proble as Nonconvex Quadratically Constrained Program
% (QCP), dubbed ReLU-QCP, that is,
%  Find W in [-M,M]^(n x r), H in [-M,M]^(r x n) such that
%        (W*H)_ii  = 1
%        (W*H)_ij <= 0,  i ~= j
%  i.e. max(0,W*H) = I.  Gurobi solves the nonconvex QCQP.
% 
% The identity matrix of any dimension can be decomposed by an RMD model
% max(0,WH) of rank exactly three. Nevertheless, computing the
% factorization in practice is challenging. We show that by solving the
% formulation above we can achieve this task.
%
% This script computes rank-r RMD of the identy matrix using YALMIP with
% gourobi soubroutine. It also plots the configuration of points and
% hyperplanes associated to the factors of the decomposition.
% ================================================================

clear all;
close all
clc;

%% ================================================================
% PARAMETERS
% ================================================================
addpath('C:\gurobi1302\win64\matlab')
addpath(genpath("C:\Users\Giovanni\Desktop\Matlab_packages\YALMIP-master"))
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'Algorithms_and_utils'));

sizes = 4:10;      % matrix dimensions
r     = 3;         % fixed factorization rank
nrep  = 1;         % repetitions
M     = 10;        % bounds on entries of W and H

rng(2);

% YALMIP / Gurobi options
ops = sdpsettings('solver','gurobi', 'verbose',0, 'cachesolvers',1, ...
    'gurobi.NonConvex',2);

results = table();

%% ================================================================
% MAIN LOOP
% ================================================================

for n = sizes
    for rep = 1:nrep
        %rng(rep)
        fprintf('\n====================================================\n');
        fprintf('Dimension: %d x %d   Rank: %d   Repetition: %d\n', n,n,r,rep);
        fprintf('====================================================\n');

        %% 1. X = IDENTITY
        X   = eye(n);
        pos = X > 0;

        fprintf('Number of positive entries in X: %d / %d\n', nnz(pos), n*n);
        fprintf('Sparsity of X: %.2f %%\n', 100*nnz(~pos)/numel(X));

        %% 2. YALMIP MODEL
        yalmip('clear');                     % free variables of previous runs
        Wv = sdpvar(n,r,'full');
        Hv = sdpvar(r,n,'full');
        P  = Wv*Hv;                          % bilinear expression

        F = [-M <= Wv(:) <= M, ...
             -M <= Hv(:) <= M, ...
             P(pos)  == X(pos), ...          % (WH)_ii = 1
             P(~pos) <= 0];                  % (WH)_ij <= 0, i ~= j

        %% 3. SOLVE (empty objective: feasibility problem)
        tic;
        sol = optimize(F, [], ops);
        elapsed_time = toc;                  % YALMIP modeling + solve

        %% 4. EXTRACT SOLUTION
        W = value(Wv);
        H = value(Hv);

        if all(isfinite([W(:); H(:)]))
            X_rec     = max(0, W*H);
            rel_error = norm(X - X_rec,'fro') / max(norm(X,'fro'),eps);
        else
            W = [];  H = [];
            rel_error = NaN;
        end

        %% 5. DISPLAY
        fprintf('YALMIP status      : %s\n', sol.info);
        fprintf('Solver time        : %.6f seconds\n', sol.solvertime);
        fprintf('Total time         : %.6f seconds\n', elapsed_time);
        fprintf('Relative error     : %.3e\n', rel_error);

        %% 6. SAVE RESULTS
        newrow = table(n, r, rep, sol.solvertime, elapsed_time, rel_error, ...
            string(sol.info), ...
            'VariableNames', {'Dimension','Rank','Repetition', ...
            'SolverTime','TotalTime','RelativeError','Status'});

        results = [results; newrow]; %#ok<AGROW>
    end
end

%% ================================================================
% FINAL TABLE
% ================================================================

fprintf('\n====================================================\n');
fprintf('FINAL RESULTS\n');
fprintf('====================================================\n');
disp(results);

%Uncomment if you want to save the results
%Save results if needed (last result is saved)
%save('gurobi_identity_WH.mat', 'W', 'H');


%% ================================================================
% 3D GEOMETRIC REPRESENTATION OF W AND H
%
% Rows w_i of W are points in R^3; each column h_j of H defines the
% plane  gamma_j = {x : h_j' x = 0}  through the origin.
% Since max(0,W*H) = I:
%     h_j' w_j  = 1 > 0   (w_j is on the positive side of gamma_j)
%     h_j' w_i <= 0       (all other points are on the other side)
%
% Two planes are shown, chosen so that their normals are as far as
% possible from parallel. The camera looks (almost) along their
% intersection line, so both planes appear nearly edge-on.
% ================================================================

%% USER PARAMETERS
selected_planes = [];    % e.g. [1 3] to choose by hand; [] = automatic
tilt            = 0.35;  % 0 = planes exactly edge-on; larger = more 3D

%% CHECKS
if isempty(W) || isempty(H)
    error('W and H must be available in the workspace.');
end
[n,r] = size(W);
if r ~= 3 || size(H,1) ~= 3
    error('W must be n x 3 and H must be 3 x n.');
end

relative_error = norm(X - max(0,W*H),'fro') / max(norm(X,'fro'),eps);

fprintf('\n====================================================\n');
fprintf('3D GEOMETRIC REPRESENTATION (n = %d)\n', n);
fprintf('====================================================\n');
fprintf('Relative ReLU reconstruction error: %.3e\n', relative_error);

%% SELECT TWO WELL-SEPARATED PLANES
hn = vecnorm(H);                         % norms of the columns of H

if isempty(selected_planes)
    Hn = H ./ max(hn, eps);
    C  = abs(Hn' * Hn);                  % |cos| of angles between normals
    C(1:n+1:end) = Inf;                  % exclude j1 = j2
    bad = hn < 1e-12;                    % exclude (near) zero columns
    C(bad,:) = Inf;   C(:,bad) = Inf;
    [~, idx] = min(C(:));
    [j1, j2] = ind2sub([n n], idx);
    selected_planes = sort([j1 j2]);
end

if numel(selected_planes) ~= 2 || any(selected_planes < 1 | selected_planes > n)
    error('selected_planes must contain two indices between 1 and %d.', n);
end

h1 = H(:,selected_planes(1)) / hn(selected_planes(1));
h2 = H(:,selected_planes(2)) / hn(selected_planes(2));

fprintf('Selected planes: gamma_%d, gamma_%d (angle between normals: %.1f deg)\n', ...
    selected_planes(1), selected_planes(2), acosd(abs(h1'*h2)));

%% PLOT SIZE
center = mean(W,1)';                     % centroid of the points
pts    = [W; zeros(1,3)];                % points + origin
scale  = max(vecnorm(pts - center', 2, 2));
if scale < 1e-10, scale = 1; end
plane_size = 1.1*scale;

%% FIGURE
figure('Color','w');
hold on; grid on; box on;

cols = lines(2);                         % one color per plane

% Point colors: w_j has the color of gamma_j, the others are gray
pcol = repmat([0.55 0.55 0.55], n, 1);
pcol(selected_planes(1),:) = cols(1,:);
pcol(selected_planes(2),:) = cols(2,:);

scatter3(W(:,1), W(:,2), W(:,3), 80, pcol, 'filled', 'MarkerEdgeColor','k');

for i = 1:n
    text(W(i,1), W(i,2), W(i,3), sprintf('  $w_{%d}$',i), ...
        'Interpreter','latex', 'FontSize',15, 'Color','k');
end

%% PLANES
for jj = 1:2

    j = selected_planes(jj);
    h = H(:,j) / hn(j);

    % Point of gamma_j closest to the centroid (the patch is centered
    % there; the plane itself is the original one, through 0)
    p0 = center - (h'*center)*h;

    % Orthonormal basis (u,v) of gamma_j
    [~,~,V] = svd(h');
    u = V(:,2);   v = V(:,3);

    [S,T] = meshgrid(linspace(-plane_size, plane_size, 2));

    surf(p0(1) + S*u(1) + T*v(1), ...
         p0(2) + S*u(2) + T*v(2), ...
         p0(3) + S*u(3) + T*v(3), ...
         'FaceColor',cols(jj,:), 'FaceAlpha',0.15, ...
         'EdgeColor',cols(jj,:), 'EdgeAlpha',0.5);

    % Plane label at a corner of the patch
    lp = p0 + 0.85*plane_size*u + 0.85*plane_size*v;
    text(lp(1),lp(2),lp(3), sprintf('$\\gamma_{%d}$',j), ...
        'Interpreter','latex', 'FontSize',17, 'Color','k');

    % Check of the sign pattern
    s = H(:,j)' * W';
    fprintf('gamma_%d:  h_%d''w_%d = %.3e,   max_{i~=%d} h_%d''w_i = %.3e\n', ...
        j, j, j, s(j), j, j, max(s([1:j-1, j+1:n])));
end
%% AXES AND VIEW
xlabel('$x_1$','Interpreter','latex','FontSize',16);
ylabel('$x_2$','Interpreter','latex','FontSize',16);
zlabel('$x_3$','Interpreter','latex','FontSize',16);
axis equal tight;

% Look along the intersection line of the two planes, slightly tilted
d = cross(h1, h2);   d = d / norm(d);
if d'*center < 0, d = -d; end            % camera on the side of the points
p = (h1 + h2) / norm(h1 + h2);           % bisector of the normals (orthogonal to d)

view((d + tilt*p)');
camup(p');                               % planes appear symmetric about the vertical
rotate3d on;
hold off;