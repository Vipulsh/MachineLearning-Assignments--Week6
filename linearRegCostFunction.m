function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


Jlet =zeros(1,m);
Jlet=X*theta;
for i=1:m,
  J +=(Jlet(i)-y(i))^2;  
endfor

J=J/(2*m);

Jreg=0;
  for i=2:size(theta),
    Jreg += theta(i)^2;  
  endfor
  

Jreg=(lambda/(2*m))*Jreg;

J=J+Jreg;


% =========================================================================

grad = grad(:);

for i=1:size(theta),
  for j=1:m,
 
     grad(i) +=  (Jlet(j) - y(j))*X(j,i);       
 
  endfor

endfor

for i=2:size(theta),
    grad(i)+= (lambda)*theta(i);    
  endfor
  
grad=grad./m;



end
