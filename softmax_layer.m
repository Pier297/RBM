% Nice defaults:
% MAX_EPOCHS = 10;
% BATCH_SIZE = 10;
% eps = 1e-1;
function [theta, pred_Y_train, pred_Y_test] = softmax_layer(X_train, Y_train, X_test, Y_test, eps, MAX_EPOCHS, BATCH_SIZE)
    n_input = size(X_train, 2);
    n_classes = 10;
    theta = randn(n_input, n_classes);
    
    theta = SGD(X_train, Y_train, theta, eps, MAX_EPOCHS, BATCH_SIZE);
    
    train_accuracy = accuracy(X_train, Y_train, theta);
    test_accuracy = accuracy(X_test, Y_test, theta);
    fprintf('\nTrain accuracy = %.3f\n', train_accuracy)
    fprintf('Test accuracy = %.3f\n', test_accuracy)
    
    pred_Y_train = predict(X_train, theta);
    pred_Y_test = predict(X_test, theta);
    
    figure
    confusionchart(Y_train, pred_Y_train)
    sgtitle('Train Confusion Matrix')
    saveas(gcf, 'Results\train_confusion_chart.png')
    
    figure
    confusionchart(Y_test, pred_Y_test)
    sgtitle('Test Confusion Matrix')
    saveas(gcf, 'Results\test_confusion_chart.png')
end


function a = accuracy(X, Y, theta)
    m = size(X, 1); K = size(theta, 2);
    correct = 0;
    for i = 1:m
        x = X(i, :)';
        prob_k = softmax(x, theta);
        [M, I] = max(prob_k);
        if string(I-1) == Y(i)
            correct = correct + 1;
        end
    end
    a = correct / m;
end

function J = cost(X, Y, theta)
    J = 0.0;
    m = size(X, 1); K = size(theta, 2);
    for i = 1:m
        prob_k = softmax(X(i, :)', theta);
        for k = 1:K
            J = J + ((Y(i) == string(k - 1)) * log(prob_k(k)));
        end
    end
    J = -J;
end

function theta = mini_batch(X, Y, theta, eps)
    delta_theta = zeros(size(theta, 1), size(theta, 2));
    % For each data point of the mini-batch
    for k = 1:size(theta, 2)
        delta_theta(:, k) = -gradient(X, Y, theta, k);
    end
    % Update theta
    theta = theta + (eps / size(X, 1)) * delta_theta;
end

function theta = SGD(X, Y, theta, eps, MAX_EPOCHS, BATCH_SIZE)
    for t = 1:MAX_EPOCHS
        t_start = tic;
        % Shuffle the training data
        perm = randperm(size(X, 1));
        X = X(perm, :);
        Y = Y(perm, :);
        
        % Mini-Batch learning
        for i = 1:ceil(size(X, 1) / BATCH_SIZE)
            pre_index = (i - 1) * BATCH_SIZE + 1;
            batch_X = X(pre_index:min(end, i*BATCH_SIZE), :);
            batch_Y = Y(pre_index:min(end, i*BATCH_SIZE));
            theta = mini_batch(batch_X, batch_Y, theta, eps);
        end
        t_end = toc(t_start);
        % Print error
        fprintf('Epoch %d | error = %d | accuracy = %.2f | Time = %.2f seconds\n', t, cost(X, Y, theta), accuracy(X, Y, theta), t_end);
    end
end

% The j-th component of gradient basically determines
% the gradient of the j-th row of theta, while the column is given by k
function nabla_theta_k_J = gradient(X, Y, theta, k)
    nabla_theta_k_J = 0.0;
    m = size(X, 1);
    for i = 1:m
        x = X(i, :)';
        prob_k = softmax(x, theta);
        nabla_theta_k_J = nabla_theta_k_J + (x * ((Y(i) == string(k - 1)) - prob_k(k)));
    end
    nabla_theta_k_J = -nabla_theta_k_J;
end

% Returns the softmax activation of an input x for class k
% Requires:
%    x is a column vector in R^{n_input}
%    theta is in R^{n_input x n_classes}
function prob_k = softmax(x, theta)
    normalizer = 0.0;
    K = size(theta, 2);
    prob_k = zeros(K, 1);
    % Compute normalizing constant
    for k = 1:K
        prob_k(k) = exp(theta(:, k)' * x);
        normalizer = normalizer + (exp(theta(:, k)' * x));
    end
    prob_k = prob_k / normalizer;
end

function pred_Y = predict(X, theta)
    m = size(X, 1); K = size(theta, 2);
    pred_Y = zeros(size(X, 1), 1);
    for i = 1:m
        x = X(i, :)';
        prob_k = softmax(x, theta);
        [M, I] = max(prob_k);
        pred_Y(i) = I-1;
    end
    pred_Y = categorical(discretize(pred_Y, 0:1:10, 'categorical', {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}));
end