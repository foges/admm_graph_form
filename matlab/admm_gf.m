function [x, y, history] = admm_gf(A, f_prox, g_prox, fg_obj, rho)
%ADMM_GF Graph Form ADMM
%  Solves optimization problems in the form 
%
%     min f(y) + g(x)
%     st. Ax = y
%
% where the proximal operators of each component of f and g are known.

% Set constants
ABSTOL = 1e-4;
RELTOL = 1e-3;
MAXITR = 500;
QUIET  = false;

[m, n] = size(A);
is_skinny = m >= n;

% Initialize variables
x = zeros(n, 1);
y = zeros(m, 1);
xt = zeros(n, 1);
yt = zeros(m, 1);
z12_norm = inf;

% Pre-compute factorization
if is_skinny
  U = chol(eye(n) + A' * A);
else
  AAt = A * A';
  U = chol(eye(m) + AAt);
end

if ~QUIET
  fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAXITR
  % x^{k+1/2}
  x12 = g_prox(x - xt);
  y12 = f_prox(y - yt);
  
  % (x^{k+1}, y^{k+1}) = Pi_A(x^{k+1/2} + \tilde x^k, y^{k+1/2} + \tilde y^k)
  if is_skinny
    x = U \ (U' \ (x12 + xt + A' * (y12 + yt)));
    y = A * x;
  else
    y = U \ (U' \ (A * (x12 + xt) + AAt * (y12 + yt)));
    x = (x12 + xt) + A' * (y12 + yt - y);
  end
  
  % \tilde x^{k+1} = tilde x^k + x^{k+1/2} - x^k
  xt = xt + x12 - x;
  yt = yt + y12 - y;
  
  % Calculate Residual 
  z = [x; y];
  z12 = [x12; y12];
  zt = [xt; yt];
  
  history.objval(k) = fg_obj(x, y);

  history.r_norm(k) = norm(z - z12);
  history.s_norm(k) = norm(-rho * (z - z12));

  history.eps_pri(k) = sqrt(n) * ABSTOL + RELTOL * max(norm(z), z12_norm);
  history.eps_dual(k) = sqrt(n) * ABSTOL + RELTOL * norm(rho * zt);
  
  z12_norm = norm(z12);
  
  if ~QUIET
    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
        history.r_norm(k), history.eps_pri(k), ...
        history.s_norm(k), history.eps_dual(k), history.objval(k));
  end

  if (history.r_norm(k) < history.eps_pri(k) && ...
      history.s_norm(k) < history.eps_dual(k))
    break;
  end
end

