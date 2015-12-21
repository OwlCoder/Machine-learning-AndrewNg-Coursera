%In this code, I am trying to use logistic regression (linear decision
%boundary) to predict if the student is admitted or not depending on two
%features, Exam 1 and 2 scores.

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
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%`````````````````cost function`````````````````````````````````

X = [ones(m, 1) X];
theta=zeros(n+1,1);
[J,grad]=costfunction(X,Y, theta);
fprintf('Cost at initial theta (zeros): %f\n', J);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%``````using optimisation toolset(rather than chosing alpha)``````````

options=optimset('GradObj','on','MaxIter',400);
[theta, J]=fminunc(@(t)(costfunction(X,Y,t)),theta, options);
fprintf('Cost at theta found by fminunc: %f\n', J);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%``````````plot it on the same graph``````````````````````````````````

plotDecisionBoundary(theta, X, Y);

