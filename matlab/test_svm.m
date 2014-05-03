function results = test_svm(m, n, rho, quiet)
%%TEST_SVM Test ADMM on non-negative least squares.
%   Compares ADMM to CVX when solving the problem
%
%     minimize    (1/2) ||w||_2^2 +  \lambda \sum (a_i^T * [w; b] + 1)_+
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x
%
%   where g_{1..n-1}(x_i) = (1/2) * w ^ 2, 
%         g_{n}(x_i)      = 0,
%         f_(y_i)         = lambda * max(y_i + 1, 0).
%
%   Test data are generated as follows
%     - Entries in A are given by the formula -y_j * (x_j^T - 1), where x_j
%       is a "feature vector" and y_j is either -1 or +1 depending on
%       which class the j'th feature belongs to. The x_j belonging to the
%       first class are drawn from a normal distribution with mean [1..1]^T
%       and covariance equal to the identity. The x_j belonging to the
%       second class are generated in a simlar fashion, except the mean
%       is equal to [-1..-1]^T. Half the training belong to the first class
%       and the other half to the second class.
%     - The parameter lambda is set to 1.0.
%
%   results = test_svm()
%   results = test_svm(m, n, rho, quiet)
% 
%   Optional Inputs: (m, n), rho, quiet
%
%   Optional Inputs:
%   (m, n)    - (default 1000, 100) Dimensions of the matrix A.
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
%   References:
%   http://www.stanford.edu/~boyd/papers/admm/svm/linear_svm_example.html
%     Formulation was taken form this example.

% Parse inputs.
if nargin < 2
  m = 1000;
  n = 100;
elseif m < n
  error('A must be a skinny matrix')
elseif mod(m, 2) ~= 0
  m = m + 1;
end
if nargin < 3
  rho = 1.0;
end
if nargin < 4
  quiet = false;
end

% Initialize Data
rng(0, 'twister')

lambda = 1.0;
N = m / 2;

x = [randn(n, N) + ones(n, N), randn(n, N) - ones(n, N)];
y = [ones(1, N), -ones(1, N)];
A = [-((ones(n, 1) * y) .* x)', -y'];

% Declare Proximal operators
f_prox = @(x, rho) max(0, x + 1 - lambda / rho) + min(0, x + 1) - 1;
g_prox = @(x, rho) [rho * x(1:end-1) / (1 + rho); x(end)];
obj_fn = @(x, y) 1 / 2 * norm(x(1:n)) ^ 2 + lambda * sum(max(0, y + 1));

% Initialize ADMM input
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 2000;
params.RELTOL = 1e-4;
params.ABSTOL = 5e-5;

% Solve using ADMM
tic
x_admm = admm(f_prox, g_prox, obj_fn, A, params);
time_admm = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n+1)
  minimize(1 / 2 * x_cvx(1:n)' * x_cvx(1:n) + ...
      lambda * sum(max(0, A * x_cvx + 1)));
cvx_end
time_cvx = toc;

% Compute error metrics
results.rel_err_obj = ...
    (obj_fn(x_admm, A * x_admm) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_admm - x_cvx) / norm(x_cvx);
results.max_violation = nan;
results.avg_violation = nan;
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
