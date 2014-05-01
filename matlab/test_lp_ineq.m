%% Test LP in Inequality Form
%
%    min. c^T * x  s.t. A * x <= b
%
% We transform this problem to
%
%  minimize    f(y) + g(x)
%  subject to  y = A * x
%
% where g_i(x_i) = c_i * x_i,
%       f_i(y_i) = I(y_i <= b_i)
%

% Initialize Data
rho = 1;

m = 1000;
n = 100;

A = randn(m, n);
b = A * rand(n, 1) + rand(m, 1);
c = rand(n, 1);

g_prox = @(x, rho) x - c / rho;
f_prox = @(x, rho) min(b, x);
obj_fn = @(x, y) c' * x;

params.rho = rho;
params.quiet = false;
params.MAXITR = 200;

% Solve using ADMM
tic
[x, factors] = admm(f_prox, g_prox, obj_fn, A, params);
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

% Print Error Metrics
fprintf('Relative Error: (admm_optval - cvx_optval) / cvx_optval = %e\n\n', ...
    (obj_fn(x, A * x) - cvx_optval) / cvx_optval)
fprintf('Constraint Error: min(x_admm) = %e, min(x_cvx) = %e\n\n', ...
    min(b - A * x), min(b - A * x_cvx));
fprintf('Norm Difference: norm(x_admm - x_cvx): %e\n\n', norm(x - x_cvx))
fprintf('Time: ADMM %f sec, CVX %f sec\n\n\n', admm_time, cvx_time) 
