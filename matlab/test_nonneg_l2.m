%% Non-Negative Least Squares Test
%
%    1/2 ||Ax - b||_2^2 s.t. x >= 0

% Setup
rho = 13;

m = 100;
n = 1000;

A = rand(m, n);
b = rand(m, 1);
g_prox = @(x) max(x, 0);
f_prox = @(x) (x * rho + b) / (1 + rho);
fg_obj = @(x, y) norm(y) + 1e4 * max(max(-x, 0));

% Solve ADMM
tic
[x, y, history] = admm_gf(A, f_prox, g_prox, fg_obj, rho);
admm_time = toc;

% Solve CVX
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(norm(A * x_cvx - b));
  subject to
    x_cvx >= 0;
cvx_end
cvx_time = toc;

fprintf('\nNorm of Error: %e\t (ADMM %f sec, CVX %f sec)\n\n', ...
        norm(x_cvx - x), admm_time, cvx_time)
