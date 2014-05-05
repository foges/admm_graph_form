function results = test_lp_eq(m, n, rho, quiet, save_mat)
%%TEST_LP_EQ Test ADMM on an equality constrained LP.
%   Compares ADMM to CVX when solving the problem
%
%     minimize    c^T * x
%     subject to  Ax = b
%                 x >= 0.
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [A; c^T] * x,
%
%   where g(x_i)        = I(x_i >= 0)
%         f_{1..m}(y_i) = I(y_i = b_i)
%         f_{m+1}(y_i)  = y_i.
%
%   Test data are generated as follows
%     - The entries in A and c are drawn uniformly in [0, 1].
%     - To generate b, we first choose a vector v with entries drawn
%       uniformly from [0, 1], we assign b = A * v. This ensures that b is
%       in the range of A.
%
%   results = test_lp_eq()
%   results = test_lp_eq(m, n, rho, quiet, save_mat)
% 
%   Optional Inputs: (m, n), rho, quiet, save_mat
%
%   Optional Inputs:
%   (m, n)    - (default 200, 1000) Dimensions of the matrix A.
%   
%   rho       - (default 1.0) Penalty parameter to proximal operators.
% 
%   quiet     - (default false) Set flag to true, to disable output to
%               console.
%
%   save_mat  - (default false) Save data matrices to MatrixMarket files.
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
  m = 200;
  n = 1000;
elseif m > n
  error('A must be a fat matrix')
end
if nargin < 3
  rho = 1.0;
end
if nargin < 4
  quiet = false;
end
if nargin < 5
  save_mat = false;
end

% Initialize Data.
rng(0, 'twister')

A = rand(m, n);
b = A * rand(n, 1);
c = rand(n, 1);

% Export Matrices
if save_mat
  mmwrite('data/A_lp_eq.dat', A, 'Matrix A for test_lp_eq.m')
  mmwrite('data/b_lp_eq.dat', b, 'Matrix b for test_lp_eq.m')
  mmwrite('data/c_lp_eq.dat', c, 'Matrix c for test_lp_eq.m')
end

% Declare proximal operators.
g_prox = @(x, rho) max(x, 0);
f_prox = @(x, rho) [b; x(end) - 1 / rho];
obj_fn = @(x, y) c' * x;

% Initialize ADMM input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 2000;
params.RELTOL = 5e-5;

% Solve using ADMM.
tic
x_admm = admm(f_prox, g_prox, obj_fn, [A; c'], params);
time_admm = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(c' * x_cvx);
  subject to
    A * x_cvx == b;
    x_cvx >= 0;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_admm, A * x_admm) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_admm - x_cvx) / norm(x_cvx);
results.max_violation = max([abs(b - A * x_admm); max(-x_admm, 0)]);
results.avg_violation = mean([abs(b - A * x_admm); max(-x_admm, 0)]);
results.time_admm = time_admm;
results.time_cvx = time_cvx;

% Print error metrics.
if ~quiet
  fprintf('\nRelative Error of Objective: %e\n', results.rel_err_obj)
  fprintf('Relative Difference in Solution: %e\n', results.rel_diff_soln)
  fprintf('Maximum Constraint Violation: %e\n', results.max_violation)
  fprintf('Average Constraint Violation: %e\n', results.avg_violation)
  fprintf('Time ADMM: %e\n', results.time_admm)
  fprintf('Time CVX: %e\n', results.time_cvx)
end

end
