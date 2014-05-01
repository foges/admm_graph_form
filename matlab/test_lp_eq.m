%% Test LP in Equality Form 
%
%    min. c^T * x  s.t. A * x = b, x >= 0
%
% We transform this problem to
%
%  minimize    f(y) + g(x)
%  subject to  y = [A; c^T] * x
%
% where g(x_i)        = I(x_u >= 0),
%       f_{1..m}(y_i) = I(y_i = b_i),
%       f_{m+1}(y_i)  = y_i
%

% Initialize Data
rho = 1.0;

m = 100;
n = 1000;

A = rand(m, n);
b = A * rand(n, 1);
c = rand(n, 1);

g_prox = @(x, rho) max(x, 0);
f_prox = cell(m, 1);
for i = 1:m
  f_prox{i} = @(x, rho) b(i);
end
f_prox{m+1} = @(x, rho) x - 1 / rho;
obj_fn = @(x, y) c' * x;

params.rho = rho;
params.quiet = true;
params.MAXITR = 1000;

% Solve using ADMM
tic
[x, factors] = admm(f_prox, g_prox, obj_fn, [A; c'], params);
admm_time = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(c' * x_cvx);
  subject to
    A * x_cvx == b;
    x_cvx >= 0;
cvx_end
cvx_time = toc;

% Print Error Metrics
fprintf('Relative Error: (admm_optval - cvx_optval) / cvx_optval = %e\n\n', ...
    (obj_fn(x, A * x) - cvx_optval) / cvx_optval)
fprintf('Constraint Error: max(abs(b - A * x_admm)) = %e, max(abs(b - A * x_cvx)) = %e\n\n', ...
    max(abs(b - A * x)), max(abs(b - A * x_cvx)));
fprintf('Norm Difference: norm(x_admm - x_cvx): %e\n\n', norm(x - x_cvx))
fprintf('Time: ADMM %f sec, CVX %f sec\n\n\n', admm_time, cvx_time) 
