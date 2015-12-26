function theta_all=onevsall(X, y, num_classes, lambda)
[m,n]=size(X);
theta_all=zeros(num_classes, n)

options=optimset('GradObj','on', 'MaxIter',40);
for c=1:num_classes
    initial_theta=theta_all(c,:)';
    [theta_all(c,:)] = fmincg (@(t)(CFReg_vectorized(X, (y == c),t, lambda)), initial_theta, options);
end
end