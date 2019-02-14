function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

X1 = X(:, 1);
x2 = X(:, 2);

params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

t = zeros(64, 1);

pairs = zeros(64, 2);
for i = 1 : size(params, 2),

    for j = 1 : size(params, 2),
        pairs((i - 1) * 8 + j,:) = [params(i), params(j)];
    end;

end;

for i = 1 : size(pairs, 1),

    C = pairs(i, 1);
    sigma = pairs(i, 2);

    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    
    predictions = svmPredict(model, Xval);
    t(i) = mean(double(predictions ~= yval));
end;

[min, minIndex] = min(t);

C = pairs(minIndex, 1);
sigma = pairs(minIndex, 2);

% =========================================================================

end
