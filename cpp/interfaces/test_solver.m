%% Lasso Example 

% Definitions
kAbs = 0; kHuber = 1; kIdentity = 2; kIndBox01 = 3; kIndEq0 = 4;
kIndGe0 = 5; kIndLe0 = 6; kLogistic = 7; kNegLog = 8; kMaxNeg0 = 9;
kMaxPos0 = 10; kSquare = 11; kZero = 12;

% Setup
n = 200;
m = 1000;

A = 1 / n * rand(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.5 * randn(m, 1);
lambda = 9e-2;

f.f = kSquare * ones(m, 1);
f.b = b;
g.f = kAbs * ones(n, 1);
g.c = lambda * ones(n, 1);

% Solve
tic
[x, y] = solver(A, f, g);
admm_time = toc;

tic
cvx_begin
  variable x_cvx(n)
  minimize(1 / 2 * sum_square_abs(A * x_cvx - b) + lambda * norm(x_cvx, 1))
cvx_end
cvx_time = toc;

fprintf('admm_optval: %e, admm_time: %e\n', ...
        1 / 2 * norm(A * x - b) ^ 2 + lambda * norm(x, 1), admm_time);
fprintf('cvx_optval:  %e, cvx_time:  %e\n', ...
        1 / 2 * norm(A * x_cvx - b) ^ 2 + lambda * norm(x_cvx, 1), cvx_time);

