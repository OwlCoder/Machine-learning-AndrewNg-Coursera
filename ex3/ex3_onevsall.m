clear all;
close all;
clc;

%```````loading the .mat data`````````````````
load('ex3data1.mat');

%``````````````display sampled data````````````````
[m,n]=size(X);      %m=5000, n=400
rand_indices=randperm(m, 100);
sel=X(rand_indices,:);
displayData(sel);
fprintf('Program paused. Press enter to continue.\n');
pause;

%````````````vectorized cost function``````````````````
lambda=0.1
theta=zeros(n+1,1)
theta_all=onevsall(X, y, theta, lambda) %theta_all is k x n+1
predictions = sigmoid(X * theta);
J = (1 / m) * (sum(-y .* log(predictions) - (1 - y) .* log(1 - predictions))) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

%