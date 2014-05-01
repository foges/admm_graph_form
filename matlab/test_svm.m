%% Test LP in Equality Form 
%
%    min.  (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+
%
% We transform this problem to
%
%  minimize    f(y) + g(x)
%  subject to  y = A * x
%
% where g_{1..n}(x_i) = (1/2) * w ^ 2, 
%       g_{n+1}(x_i)  = 0,
%       f_(y_i)       = lambda * max(y_i + 1, 0)
%

% Initialize Data
% Formulation taken from: 
%   http://www.stanford.edu/~boyd/papers/admm/svm/linear_svm_example.html
rng(0, 'twister')

rho = 1;
lambda = 1.0;

n = 100;
m = 1000;
N = m / 2;
M = m / 2;

% Positive examples
Y = randn(n, N) + ones(n, N);

% Negative examples
X = randn(n, M) - ones(n, M);

x = [X, Y];
y = [ones(1, N), -ones(1, M)];
A = [ -((ones(n, 1) * y) .* x)', -y'];

f_prox = @(x, rho) max(0, x + 1 - lambda / rho) + min(0, x + 1) - 1;
g_prox = cell(n + 1, 1);
for i = 1:n
  g_prox{i} = @(x, rho) rho * x / (1 + rho);
end
g_prox{n+1} = @(x, rho) x;
obj_fn = @(x, y) 1 / 2 * norm(x(1:n))^2 + ...
    lambda * sum(max(0, A * x + 1));

params.rho = rho;
params.quiet = true;
params.MAXITR = 1000;

% Solve using ADMM
tic
[x, factors] = admm(f_prox, g_prox, obj_fn, A, params);
admm_time = toc;

% Solve using CVX
tic
cvx_begin quiet
  variable x_cvx(n+1)
  minimize(1 / 2 * x_cvx(1:n)' * x_cvx(1:n) + ...
      lambda * sum(max(0, A * x_cvx + 1)));
cvx_end
cvx_time = toc;

% Print Error Metrics
fprintf('Relative Error: (admm_optval - cvx_optval) / cvx_optval = %e\n\n', ...
    (obj_fn(x, A * x) - cvx_optval) / cvx_optval)
fprintf('Norm Difference: norm(x_admm - x_cvx): %e\n\n', norm(x - x_cvx))
fprintf('Time: ADMM %f sec, CVX %f sec\n\n\n', admm_time, cvx_time) 
