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
Test_Set = [0.01 0.03 0.1 0.3 1, 3, 10 30];
steps = length(Test_Set);

Optimize_Matrix = [];

for i=1:steps
    for j=1:steps
        
        c_temp = Test_Set(i);
        sigma_temp = Test_Set(j);
        
        model = svmTrain(X, y, c_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        
        Optimize_Matrix = [Optimize_Matrix; c_temp, sigma_temp, prediction_error];
    end
end

[Error, Index] = min(Optimize_Matrix(:,3));


C = Optimize_Matrix(Index, 1);
sigma = Optimize_Matrix(Index, 2);


% =========================================================================

end
