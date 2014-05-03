function results = test_nonneg_l2(m, n, rho, quiet)
%%TEST_NONNEG_L2 Test ADMM on non-negative least squares.
%   Compares ADMM to CVX when solving the problem
%
%     minimize    (1/2) ||Ax - b||_2^2 
%     subject to  x >= 0
%
%   We transform this problem into
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x
%
%   where g_i(x_i) = I(x_i >= 0),
%         f_i(y_i) = (1/2) * (y_i - b_i) ^ 2.
%
%   Test data are generated as follows
%     - Entries in A are generated uniformly at random in [0, 1/n].
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
%                   (admm_optval - cvx_optval) / abs(cvx_optval).
%                 + rel_err_soln: Relative difference in solution between
%                   CVX and ADMM, defined as 
%                   norm(x_admm - x_cvx) / norm(x_cvx).
%                 + max_violation: Maximum constraint violation (nan if
%                   problem has no constraints).
%                 + avg_violation: Average constraint violation.
%                 + time_admm: Time required by ADMM to solve problem.
%                 + time_cvx: Time required by CVX to solve problem.
%

% Parse inputs.
if nargin < 2
  m = 1000;
  n = 200;
end
if nargin < 3
  rho = 1.0;
end
if nargin < 4
  quiet = false;
end

% Initialize Variables
rng(0, 'twister')

n_half = floor(2 * n / 3);
n_2ndhalf = n - n_half;
A = 1 / n * rand(m, n);
b = A * [ones(n_half, 1); -ones(n_2ndhalf, 1)]  + 0.01 * randn(m, 1);

% Declare Proximal operators
g_prox = @(x, rho) max(x, 0);
f_prox = @(x, rho) (x * rho + b) / (1 + rho);
obj_fn = @(x, y) 1 / 2 * norm(A * x - b) ^ 2;

% Initialize ADMM input
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
params.RELTOL = 1e-4;

% Solve using ADMM
tic
x_admm = admm(f_prox, g_prox, obj_fn, A, params);
time_admm = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(1/2 * (A * x_cvx - b)' * (A * x_cvx - b));
  subject to
    x_cvx >= 0;
cvx_end
time_cvx = toc;

% Compute error metrics
results.rel_err_obj = ...
    (obj_fn(x_admm, A * x_admm) - cvx_optval) / cvx_optval;
results.rel_diff_soln = norm(x_admm - x_cvx) / norm(x_cvx);
results.max_violation = abs(min(min(x_admm), 0));
results.avg_violation = mean(abs(min(x_admm, 0)));
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
