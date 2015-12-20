function [J,grad]=costfunction(X,y, theta)
[m,n]=size(X);
grad=zeros(n,1);
predictions=sigmoid(X*theta);
J = 1 / m * sum(-y .* log(predictions) - (1 - y) .* log(1 - predictions));
for i = 1:n
    grad(i) = 1 / m * sum((predictions - y) .* X(:, i));
end
end