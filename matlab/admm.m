function [x12, factors] = admm(prox_f, prox_g, obj_fn, A, params, factors)
% Generic graph projection splitting solver. Solve
%
%  minimize    f(y) + g(x)
%  subject to  y = Ax
% 
% given prox_f, prox_g, and A.

% Parse Input.
if nargin < 5
  params = [];
end
if nargin < 6
  factors = [];
end

ABSTOL = get_or_default(params, 'ABSTOL', 1e-4);
RELTOL = get_or_default(params, 'RELTOL', 1e-3);
MAXITR = get_or_default(params, 'MAXITR', 10000);
rho    = get_or_default(params, 'rho', 1);
quiet  = get_or_default(params, 'quiet', false);

L = get_or_default(factors, 'L', []);
D = get_or_default(factors, 'D', []);
P = get_or_default(factors, 'P', []);

% Variables.
total_time = tic;

[m, n] = size(A);
x = zeros(n, 1);     xt = zeros(n, 1);
y = zeros(m, 1);     yt = zeros(m, 1);
z = zeros(n + m, 1); zt = zeros(n + m, 1);

% Precompute AAt or AtA.
if isempty(factors) && ~issparse(A)
  if m < n
    AA = A * A';
  else
    AA = A' * A;
  end
end

if ~quiet
  fprintf('iter :\t%8s\t%8s\t%8s\t%8s\t%8s\n', 'r', 'eps_pri', 's', ...
      'eps_dual', 'objective');
end

for iter = 1:MAXITR
  %  x^{k+1/2} = prox(x^k - \tilde x^k)
  y12 = eval_prox(prox_f, y - yt, rho);
  x12 = eval_prox(prox_g, x - xt, rho);
  z12 = [x12; y12];

  zprev = z; 

  if iter == 1
    factor_time = tic;
  end

  % (x^{k+1}, y^{k+1}) = Pi_A(x^{k+1/2} + \tilde x^k, y^{k+1/2} + \tilde y^k)
  if issparse(A)
    [z, L, D, P] = project_graph(z12 + zt, A, [], L, D, P);
  else
    [z, L] = project_graph(z12 + zt, A, AA, L);
  end

  if iter == 1
    factor_time = toc(factor_time);
  end

  x = z(1:n);
  y = z(n + 1:n + m);

  % Check Optimality Conditions.
  eps_pri  = sqrt(n) * ABSTOL + RELTOL * max(norm(z12), norm(z));
  eps_dual = sqrt(n) * ABSTOL + RELTOL * norm(rho * zt);
  prires = norm(z12 - z);
  duares = rho * norm(z - zprev);

  if ~quiet && (iter == 1 || mod(iter, 10) == 0)
    obj = obj_fn(x, y);
    fprintf('%4d :\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n', ...
        iter, prires, eps_pri, duares, eps_dual, obj);
  end

  if iter > 2 && prires < eps_pri && duares < eps_dual
    break
  end

  xt = xt + x12 - x;
  yt = yt + y12 - y;
  zt = [xt; yt];
end

factors.L = L;
factors.D = D;
factors.P = P;

if ~quiet
  fprintf('factorization time: %.2e seconds\n', factor_time);
  fprintf('total iterations: %d\n', iter);
  fprintf('total time: %.2f seconds\n', toc(total_time));
end

end

function y = eval_prox(f_prox, x, rho)
% Evaluates the proximal operator(s) on x.  f_prox may either be a 
% function handle or a cell array of function handles

if iscell(f_prox)
  y = nan(size(x));
  for i = 1:length(f_prox)
    y(i) = f_prox{i}(x(i), rho);
  end
else
  y = f_prox(x, rho);
end

end

function varargout = project_graph(v, A, AA, L, D, P)
% Project v onto the graph of A. 
% Supports factorization caching and both dense/sparse A.

[m, n] = size(A);
c = v(1:n);
d = v(n + 1:end);

if issparse(A)
  if isempty(P) || isempty(L) || isempty(D)
    % Solve KKT system.
    K = [ speye(n) A' ; A -speye(m) ];
    [L, D, P] = ldl(K);
  end

  z = P * (L' \ (D \ (L \ (P' * sparse([ c + A' * d ; zeros(m, 1) ])))));

  varargout(1) = {z};
  varargout(2) = {L};
  varargout(3) = {D};
  varargout(4) = {P};
else
  if m < n
    if isempty(L)
      L = chol(eye(m) + AA);
    end
    y = L \ (L' \ (A * c + AA * d));
    x = c + A' * (d - y);
  else
    if isempty(L)
      L = chol(eye(n) + AA);
    end
    x = L \ (L' \ (c + A' * d));
    y = A * x;
  end

  varargout(1) = {[x; y]};
  varargout(2) = {L};
end

end

function output = get_or_default(input, var, default)

if isfield(input, var)
  output = input.(var);
else
  output = default;
end

end
