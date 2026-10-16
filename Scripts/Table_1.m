% ================================================================
% Script supporting the contribution
%     "The ReLU Matrix Decomposition: NP-Hardness, Ill-Posedness, and ReLU-Rank"  
% authors: Giovanni Seraghiti, Arnaud Vandaele, and Nicolas Gillis
% ================================================================
%
%  Comparison: Gurobi (via YALMIP) vs Naive-NMD for X = max(0,W*H)
%  for several values of the latent rank r
%
%  The true W and H are generated randomly and then hidden.
%  Both methods receive only X and the rank r.
%
%  Gurobi solves the feasibility problem
%
%       find  W in [-M,M]^{m x r},  H in [-M,M]^{r x n}
%       s.t.  (W*H)_ij = X_ij   if X_ij > 0
%             (W*H)_ij <= 0     if X_ij = 0
%
%  STOPPING CRITERION (same target for both methods)
%
%  Both methods aim at a relative error
%
%       ||max(0,Y) - X||_F / ||X||_F <= epsRel.
%
%  - Naive-NMD stops when its relative error falls below epsRel.
%  - Gurobi stops at the first point violating every constraint by at
%    most delta (FeasibilityTol). Each entry error is then <= delta, so
%       ||max(0,WH) - X||_F <= sqrt(m*n)*delta,
%    and the choice  delta = epsRel*||X||_F/sqrt(m*n)  guarantees
%    a relative error <= epsRel.
% ================================================================

clear; clc; close all;

addpath('C:\gurobi1302\win64\matlab')
addpath(genpath("C:\Users\Giovanni\Desktop\Matlab_packages\YALMIP-master"))
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'Algorithms_and_utils'));

%% ================================================================
% 1. PARAMETERS
% ================================================================

ranks = [5 10 15];          % latent ranks (combinations with min(m,n) <= r are skipped)

sizes = [
      25 20
      30 25
%      40 35
%      50 45
%      60 50
 %     70 60
];

nrep = 1;                  % random matrices for each (rank, size)

% Target relative error, shared by both methods
epsRel = 1e-9;

% Time limit (s), shared by both methods
timeLimit = 300;

% Generation of W,H and corresponding bound M for Gurobi.
% M must be large enough not to exclude feasible factorizations.

gen = 2;                    % 1: randn,  2: uniform in [-0.5,0.5]
if gen == 1, M = 10; else, M = 0.5; end

% Naive-NMD parameters.
param = struct('maxit',1e6, 'tol',epsRel, 'tolerr',-1, ...
               'time',timeLimit, 'display',1);

% YALMIP / Gurobi options (FeasibilityTol is set for each run)
ops = sdpsettings('solver','gurobi', 'verbose',1, ...
    'gurobi.NonConvex',2, ...
    'gurobi.TimeLimit',timeLimit);

%% ================================================================
% 2. MAIN EXPERIMENT
% ================================================================

R = struct([]);             % one entry per run, converted to a table at the end

for r = ranks
    for s = 1:size(sizes,1)

        m = sizes(s,1);  n = sizes(s,2);

        if min(m,n) <= r
            fprintf('\nSkipping %d x %d with rank %d (min(m,n) <= r)\n',m,n,r);
            continue;
        end

        fprintf('\n############ Dimension %d x %d    Rank %d ############\n',m,n,r);

        for rep = 1:nrep

            fprintf('\n------ Rank %d - %d x %d - repetition %d/%d ------\n',r,m,n,rep,nrep);

            %% ---- Generate hidden W, H and the data X ----
            rng(rep);
            if gen == 1
                Wtrue = randn(m,r);       Htrue = randn(r,n);
            else
                Wtrue = rand(m,r) - 0.5;  Htrue = rand(r,n) - 0.5;
            end
            X     = max(0, Wtrue*Htrue);
            normX = norm(X,'fro');
            pos   = X > 0;

            fprintf('nnz(X) = %d / %d (%.2f%%)\n', nnz(X), numel(X), 100*nnz(X)/numel(X));

            %% ---- Gurobi via YALMIP ----
            fprintf('\nRunning Gurobi (YALMIP)...\n');

            % Feasibility tolerance guaranteeing relative error <= epsRel,
            % clipped to the range accepted by Gurobi
            delta  = min(max(epsRel*normX/sqrt(m*n), 1e-9), 1e-2);
            opsRun = sdpsettings(ops, 'gurobi.FeasibilityTol', delta);

            yalmip('clear');                         % free variables of previous runs
            W = sdpvar(m,r,'full');
            H = sdpvar(r,n,'full');
            P = W*H;                                 % bilinear expression

            F = [-M <= W(:) <= M, -M <= H(:) <= M, P(pos) == X(pos)];
            if any(~pos(:))
                F = [F, P(~pos) <= 0];
            end

            tic;
            sol = optimize(F, [], opsRun);           % empty objective: feasibility
            gurobi_total = toc;                      % model building + solve

            % Error recomputed from the returned point.
            % No feasible point -> YALMIP returns NaN (error NaN) or zeros (error 1).
            Wg = value(W);  Hg = value(H);
            gurobi_error = NaN;
            gurobi_viol  = NaN;
            if all(isfinite([Wg(:); Hg(:)]))
                Y = Wg*Hg;
                gurobi_error = norm(max(0,Y) - X,'fro') / normX;
                % maximum constraint violation, to compare with delta
                gurobi_viol  = max([abs(Y(pos) - X(pos)); max(0, Y(~pos)); 0]);
            end

            %% ---- Naive-NMD ----
            fprintf('\nRunning Naive-NMD...\n');
            
            %starting point
            if gen==1
                W0 = randn(m,r);       H0 = randn(r,n);
                param.Theta0=W0*H0;
            else
                W0 = rand(m,r)-0.5;     H0 = rand(r,n)-0.5;
                param.Theta0=W0*H0;
            end

            [Theta,~,~,time] = Naive_NMD(X,r,param);
            naive_time  = time(end);
            naive_error = norm(max(0,Theta) - X,'fro') / normX;   % same measure as Gurobi

            %% ---- Store and print ----
            R(end+1).Rank        = r;                %#ok<SAGROW>
            R(end).m             = m;
            R(end).n             = n;
            R(end).Repetition    = rep;
            R(end).Delta         = delta;
            R(end).GurobiError   = gurobi_error;
            R(end).GurobiMaxViol = gurobi_viol;
            R(end).GurobiTime    = sol.solvertime;   % solver time only (fair comparison)
            R(end).GurobiTotal   = gurobi_total;     % including YALMIP modeling time
            R(end).GurobiStatus  = string(sol.info);
            R(end).GurobiSuccess = gurobi_error <= epsRel;   % false also if NaN
            R(end).NaiveError    = naive_error;
            R(end).NaiveTime     = naive_time;
            R(end).NaiveSuccess  = naive_error <= epsRel;

            fprintf('\n=========== RESULTS ===========\n');
            fprintf('Gurobi status:    %s\n', sol.info);
            fprintf('Gurobi delta:     %.2e   max violation: %.2e\n', delta, gurobi_viol);
            fprintf('Gurobi time:      %.4e s  (total with YALMIP: %.4e s)\n', sol.solvertime, gurobi_total);
            fprintf('Gurobi error:     %.4e   success: %d\n', gurobi_error, R(end).GurobiSuccess);
            fprintf('Naive-NMD time:   %.4e s\n', naive_time);
            fprintf('Naive-NMD error:  %.4e   success: %d\n', naive_error, R(end).NaiveSuccess);
        end
    end
end

Results = struct2table(R);

%% ================================================================
% 3. TABLES (one block per rank)
% ================================================================

displayByRank(Results, ranks, 'COMPLETE RESULTS');

% Means over repetitions:
%  - mean error over all runs that returned a point;
%  - mean time over successful runs only;
%  - number of failures (error > epsRel or no point returned).
[G, Rk, Mk, Nk] = findgroups(Results.Rank, Results.m, Results.n);
gmean = @(v) splitapply(@(x) mean(x,'omitnan'), v, G);

failG = ~Results.GurobiSuccess;
failN = ~Results.NaiveSuccess;
tG = Results.GurobiTime;  tG(failG) = NaN;
tN = Results.NaiveTime;   tN(failN) = NaN;

MeanResults = table(Rk, Mk, Nk, ...
    gmean(Results.GurobiError), gmean(tG), splitapply(@sum, failG, G), ...
    gmean(Results.NaiveError),  gmean(tN), splitapply(@sum, failN, G), ...
    'VariableNames', {'Rank','m','n', ...
    'GurobiErrorMean','GurobiTimeMean','GurobiFail', ...
    'NaiveErrorMean','NaiveTimeMean','NaiveFail'});

displayByRank(MeanResults, ranks, ...
    sprintf('MEAN RESULTS (time over successful runs, error <= %.0e) AND FAILURES', epsRel));

% Uncomment if you want to save the results
% writetable(Results,     'Gurobi_vs_NaiveNMD_results.csv');
% writetable(MeanResults, 'Gurobi_vs_NaiveNMD_mean_results.csv');

%% ================================================================
% 4. PLOTS: one curve per (method, rank)
% ================================================================

% Means arranged as (rank x size) matrices, NaN for skipped combinations
nr = numel(ranks);  ns = size(sizes,1);
[~, ir] = ismember(MeanResults.Rank, ranks);
[~, is] = ismember([MeanResults.m MeanResults.n], sizes, 'rows');
idx  = sub2ind([nr ns], ir, is);
toGrid = @(v) reshape(accumarray(idx, v, [nr*ns 1], [], NaN), nr, ns);

xlabels = compose('%d \\times %d', sizes(:,1), sizes(:,2));

makePlot(xlabels, toGrid(MeanResults.GurobiErrorMean), toGrid(MeanResults.NaiveErrorMean), ...
    ranks, 'Final relative error', 'Final reconstruction error');

makePlot(xlabels, toGrid(MeanResults.GurobiTimeMean), toGrid(MeanResults.NaiveTimeMean), ...
    ranks, 'Computational time (s)', 'Computational time (successful runs)');

%% ================================================================
% LOCAL FUNCTIONS
% ================================================================

function displayByRank(Tab, ranks, titleStr)
% Display a table split into blocks with the same rank, one below the other.
    fprintf('\n============================================================\n');
    fprintf('%s\n', titleStr);
    fprintf('============================================================\n');
    for r = ranks
        sub = Tab(Tab.Rank == r, :);
        if ~isempty(sub)
            fprintf('\n------------------------ Rank r = %d ------------------------\n', r);
            disp(sub);
        end
    end
end

function makePlot(xlabels, YG, YN, ranks, ylab, ttl)
% Gurobi: solid line with circles; Naive-NMD: dashed line with squares.
% Same color for the same rank.
    figure('Color','w');
    colors = lines(numel(ranks));
    x = 1:numel(xlabels);
    leg = cell(1, 2*numel(ranks));
    for k = 1:numel(ranks)
        semilogy(x, YG(k,:), '-o',  'Color',colors(k,:), 'LineWidth',1.5, 'MarkerSize',7); hold on;
        semilogy(x, YN(k,:), '--s', 'Color',colors(k,:), 'LineWidth',1.5, 'MarkerSize',7);
        leg{2*k-1} = sprintf('Gurobi, r = %d',    ranks(k));
        leg{2*k}   = sprintf('Naive-NMD, r = %d', ranks(k));
    end
    grid on; box on;
    xticks(x); xticklabels(xlabels);
    xlabel('Matrix dimension'); ylabel(ylab); title(ttl);
    legend(leg, 'Location','best', 'FontSize',11);
    set(gca, 'FontName','Times New Roman', 'FontSize',12);
end
