clear all;
close all;
clc;

%`````````````````````````````load data``````````````````````````````

data=csvread('ex1data2.txt');
X=data(:,(1:2));
Y=data(:,3);
[m,n]=size(X);

%`````````````````````feature normalizing````````````````````````````

mu=mean(X);
sigma=std(X);
for i=1:n
    X_norm(:,i)=(X(:,i)-mu(i))/sigma(i);
end

%````````````````````calculate predictions and cost````````````````

X=[ones(m,1), X_norm];
theta=zeros(n+1,1);
predictions=X*theta;
J=(1/(2*m))*(sum((predictions-Y).^2));

%```````````````````gradient descent``````````````````````````````````

alpha=0.01;
numIters=100;
for i=1:numIters
   predictions = X * theta;
   for i = 1:n+1
        theta(i) = theta(i) - (alpha / m) * sum((predictions - Y) .* X(:,i));
   end 
end

%`````````````````cost at this and prediction``````````````````````````

fprintf('theta0=%f, \ntheta1=%f, \ntheta2=%f \n', theta(1), theta(2), theta(3)) 
J=(1/(2*m))*(sum((predictions-Y).^2));
fprintf('cost at the above parameters is equal to %f \n', J)
fprintf('Program paused. Press enter to continue.\n');
pause;
x_eg=[1650 3];
%normalizing
x_eg(1)=(x_eg(1)-mu(1))/sigma(1);
x_eg(2)=(x_eg(2)-mu(2))/sigma(2);
x_eg=[1 x_eg];
price=x_eg*theta;
fprintf('predicting for 1650 sqft, 3 bedroom house: %f \n',price)
fprintf('Program paused. Press enter to continue.\n');
pause;

%``````````````````````normal equation`````````````````````````````````

data1=csvread('ex1data2.txt');
X=data1(:,1:2);
Y=data1(:,3);
m=length(Y);

%`````````````````no need to normalize````````````````````````````````

X=[ones(m,1) X];
theta=zeros(size(X,2),1);
theta = pinv(X' * X) * X' * Y;
x_eg=[1 1650 3];
price=x_eg*theta;
fprintf('predicting for 1650 sqft, 3 bedroom house: %f \n',price)