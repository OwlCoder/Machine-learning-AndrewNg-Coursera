clear all;
close all;
clc;

%```````loading the .mat data`````````````````

load('ex3data1.mat');
num_classes=10;
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
X=[ones(m,1),X]
theta_all=onevsall(X, y, num_classes, lambda) %theta_all is k x n+1

%```````````