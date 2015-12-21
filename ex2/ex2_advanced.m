%In this code, I am trying to use logistic regression (non-linear decision
%boundary) to predict if the student is admitted or not depending on two
%features, Exam 1 and 2 scores.

%Error: As I am settting theta equal to all zeroes, it is getting stuck in
%a local minima and hence theta is not changing and staying at zero. (there
%might be some other problem also, need to check.

close all;
clear all;
clc;

%````````````````````loading data`````````````````````````````

data=csvread('ex2data1.txt');
X=data(:,[1,2]);
Y=data(:,3);
[m,n]=size(X);

%````````````````````plot data````````````````````````````````

figure;
hold on;
pos=find(Y==1);
neg=find(Y==0);
plot(X(pos,1),X(pos,2),'k+', 'LineWidth', 2);
plot(X(neg,1),X(neg,2),'o', 'MarkerFaceColor','y');
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('admitted','not admitted');
hold on;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%`````````````````cost function`````````````````````````````````

X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

[cost, grad] = costfunction( X, Y,initial_theta);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%``````using optimisation toolset(rather than chosing alpha)``````````

options=optimset('GradObj','on','MaxIter',400,'TolX', 1e-18);
[theta, J]=fminunc(@(t)(costfunction(X,Y,t)),initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', J);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%``````````plot it on the same graph``````````````````````````````````

plotDecisionBoundary_non_linear(theta, X, Y);
%``````````NOT WORKING``````````NOT WORKING``````````NOT WORKING`````````
