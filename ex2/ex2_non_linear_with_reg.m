%In this code, I am trying to use logistic regression (non-linear decision
%boundary) to predict if the microchip is accepted or not depending on two
%features, Test 1 and 2 scores.

clear all;
close all;
clc;

%```````````````````````````plot data`````````````````````````````

data=load('ex2data2.txt');
X = data(:, [1, 2]);
Y = data(:, 3);
plotData(X, Y);
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
legend('accepted', 'not accepted');
hold on;

%```````````````````````feature mapping`````````````````````````

X = mapFeature(X(:,1), X(:,2));

%`````````````````````````initial cost````````````````````````````

initial_theta = zeros(size(X, 2), 1);
lambda = 1;
[cost, grad] = costFunctionReg(initial_theta, X, Y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%```````````````optimisation to calculate min cost`````````````````

options = optimset('GradObj', 'on', 'MaxIter', 400, 'TolX', 1e-8);
[theta, J] = fminunc(@(t)(costFunctionReg(t, X, Y, lambda)), initial_theta, options);
%```````````````plot decision boundary`````````````````````

plotDecisionBoundary_non_linear(theta, X, Y);
hold on;
title(sprintf('lambda = %g', lambda))

%````````````````calculating accuracy on the training set``````````````

p=(sigmoid(X*theta)>=0.5);
fprintf('Train Accuracy: %f\n', mean(double(p == Y)) * 100);
