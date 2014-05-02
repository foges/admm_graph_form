function results = test_lp_ineq(m, n, rho, quiet)
%%TEST_LP_INEQ Test ADMM on an inequality constrained LP.
%   Compares ADMM to CVX when solving the problem
%
%     minimize    c^T * x
%     subject to  Ax <= b
%
% We transform this problem to
%
%  minimize    f(y) + g(x)
%  subject to  y = A * x
%
% where g_i(x_i) = c_i * x_i,
%       f_i(y_i) = I(y_i <= b_i)
%
%   Test data is generated as follows
%     - Entries in A are generated normally N(0, 1/n).
%     - Entries in b are generated such that the optimal unconstrained
%       solution x^\star is approximately equal to [1..1 -1..-1]^T, 
%       guaranteeing that some constraints will be active.
%
%   results = test_nonneg_l2()
%   results = test_nonneg_l2(m, n, rho, quiet)
% 
%   Optional Inputs: (m, n), rho, quiet
%
%   Optional Inputs:
%   (m, n)    - (default 1000, 200) Dimensions of the matrix A.
%   
%   rho       - (default 1.0) Penalty parameter to proximal operators.
% 
%   quiet     - (default false) Set flag to true, to disable output to
%               console.
%
%   Outputs:
%   results   - Structure containg test results. Fields are:
%                 + rel_err_obj: Relative error of the objective, as
%                   compared to the solution obtained from CVX, defined as
%                   (admm_optval - cvx_optval) / cvx_optval.
%                 + rel_err_soln: Relative difference in solution between
%                   CVX and ADMM, define as 
%                   norm(x_admm - x_cvx) / norm(x_cvx).
%                 + max_violation: Maximum constraint violation (nan if no 
%                   problem has no constraints).
%                 + avg_violation: Average constraint violation.
%                 + time_admm: Time required by ADMM to solve problem.
%                 + time_cvx: Time required by CVX to solve problem.
%

% Parse inputs.
if nargin < 2
  m = 1000;
  n = 100;
elseif m < n
  error('A must be a skinny matrix')
end
if nargin < 3
  rho = 1.0;
end
if nargin < 4
  quiet = false;
end

% Initialize Data
rng(0, 'twister')

A = 1 / n * randn(m, n);
b = A * rand(n, 1) + 0.01 * rand(m, 1);
c = rand(n, 1);

g_prox = @(x, rho) x - c / rho;
f_prox = @(x, rho) min(b, x);
obj_fn = @(x, y) c' * x;

params.rho = rho;
params.quiet = quiet;
params.MAXITR = 200;

% Solve using ADMM
tic
x_admm = admm(f_prox, g_prox, obj_fn, A, params);
admm_time = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(c' * x_cvx);
  subject to
    A * x_cvx <= b;
cvx_end
cvx_time = toc;

% Compute error metrics
results.rel_err_obj = ...
    (obj_fn(x_admm, A * x_admm) - cvx_optval) / cvx_optval;
results.rel_diff_soln = norm(x_admm - x_cvx) / norm(x_cvx);
results.max_violation = abs(min(min(x_admm), 0));
results.avg_violation = mean(abs(x_admm(x_admm < 0)));
results.time_admm = time_admm;
results.time_cvx = time_cvx;

% Print error metrics
if ~quiet
  fprintf('\nRelative Error of Objective: %e\n', results.rel_err_obj)
  fprintf('Relative Difference in Solution: %e\n', results.rel_diff_soln)
  fprintf('Maximum Constraint Violation: %e\n', results.max_violation)
  fprintf('Average Constraint Violation: %e\n', results.avg_violation)
  fprintf('Time ADMM: %e\n', results.time_admm)
  fprintf('Time CVX: %e\n', results.time_cvx)
end

end
