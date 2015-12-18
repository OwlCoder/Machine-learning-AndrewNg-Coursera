function theta=gradientdescent(X,Y, theta, alpha, iterations)
    m=length(Y);
    for i =1:iterations
        predictions=X*theta;
        theta(1)=theta(1)-(alpha/m)*(sum((predictions-Y).*X(:,1)));
        theta(2)=theta(2)-(alpha/m)*(sum((predictions-Y).*X(:,2)));
    end
end