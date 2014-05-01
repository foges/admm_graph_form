%% Test Non-Negative Least Squares
%
%    min. (1/2) ||A * x - b||_2^2 s.t. x >= 0
%
% We transform this problem to
%
%  minimize    f(y) + g(x)
%  subject to  y = A * x
%
% where g_i(x_i) = I(x_i >= 0),
%       f_i(y_i) = (1/2) * (y_i - b_i) ^ 2
%

% Initialize Data
rng(0, 'twister')

rho = 1.0;

m = 1000;
n = 100;

A = 1 / n * rand(m, n);
b = randn(m, 1) + 1;

g_prox = @(x, rho) max(x, 0);
f_prox = @(x, rho) (x * rho + b) / (1 + rho);
obj_fn = @(x, y) 1/2 * norm(A * x - b) ^ 2;

params.rho = rho;
params.quiet = false;
params.MAXITR = 1000;

% Solve using ADMM
tic
[x, factors] = admm(f_prox, g_prox, obj_fn, A, params);
admm_time = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(1/2 * (A * x_cvx - b)' * (A * x_cvx - b));
  subject to
    x_cvx >= 0;
cvx_end
cvx_time = toc;

% Print Error Metrics
fprintf('Relative Error: (admm_optval - cvx_optval) / cvx_optval = %e\n\n', ...
    (obj_fn(x, A * x) - cvx_optval) / cvx_optval)
fprintf('Constraint Error: min(x_admm) = %e, min(x_cvx) = %e\n\n', ...
    min(x), min(x_cvx));
fprintf('Norm Difference: norm(x_admm - x_cvx): %e\n\n', norm(x - x_cvx))
fprintf('Time: ADMM %f sec, CVX %f sec\n\n\n', admm_time, cvx_time) 
