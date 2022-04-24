clear;

% Assignment 3
% 
% Implement from scratch an RBM and apply it to DSET3. The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).
% 
% 1.     Train an RBM with 100 hidden neurons (single layer) on the MNIST data (use the training set split provided by the website).
% 
% 2.     Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.
% 
% 3.     Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show the resulting confusion matrices (training and test) in your presentation.
% 
% (Alternative to step 3, optional) Step 3 can as well be realized by placing a softmax layer after the RBM: if you are up for it, feel free to solve the assignment this way.

rng(1); % fix random seed for reproducibility

% --------------- Load the MNIST dataset ---------------
[X_train, Y_train, X_test, Y_test] = read_mnist('mnist-dataset\train-images-idx3-ubyte.gz','mnist-dataset\train-labels-idx1-ubyte.gz','mnist-dataset\t10k-images-idx3-ubyte.gz','mnist-dataset\t10k-labels-idx1-ubyte.gz');


% TODO: Remove this to train on the whole dataset
% X_train = X_train(1:100, :);
% Y_train = Y_train(1:100);
% X_test = X_test(1:100, :);
% Y_test = Y_test(1:100);


% --------------- Parameters of the RBM --------------- %
n_hidden_units = 100;
k = 1;
BATCH_SIZE = 20;
eps = 0.1;
MAX_EPOCHS_RBM = 20;


MAX_EPOCHS_SOFTMAX = 20;

% --------------- Define and Train the RBM --------------- %
% fprintf('\n--- Training RBM ---\n\n')
[enc_X_train, enc_X_test, v, h, W, bias_v, bias_h, enc_rec_image_h, enc_rec_image] = RBM(X_train, X_test, n_hidden_units, k, eps, MAX_EPOCHS_RBM, BATCH_SIZE);
% save('RBM-mnist\enc_X_train', 'enc_X_train');
% save('RBM-mnist\enc_X_test', 'enc_X_test');
% save('RBM-mnist\v', 'v');
% save('RBM-mnist\h', 'h');
% save('RBM-mnist\W', 'W');
% save('RBM-mnist\bias_v', 'bias_v');
% save('RBM-mnist\bias_h', 'bias_h');

% load('RBM-mnist\enc_X_train', 'enc_X_train');
% load('RBM-mnist\enc_X_test', 'enc_X_test');
% load('RBM-mnist\v', 'v');
% load('RBM-mnist\h', 'h');
% load('RBM-mnist\W', 'W');
% load('RBM-mnist\bias_v', 'bias_v');
% load('RBM-mnist\bias_h', 'bias_h');


% ------------ Train logistic regression ----------- %
% fprintf('[]\n')
% t_start = tic;
% model = logistic_regression(enc_X_train, Y_train, enc_X_test, Y_test, []);
% t_end = toc(t_start)
% 
% fprintf('[10]\n')
% t_start = tic;
% model = logistic_regression(enc_X_train, Y_train, enc_X_test, Y_test, [10]);
% t_end = toc(t_start)
% 
% fprintf('[100]\n')
% t_start = tic;
% model = logistic_regression(enc_X_train, Y_train, enc_X_test, Y_test, [100]);
% t_end = toc(t_start)
% 
% fprintf('[100 50]\n')
% t_start = tic;
% model = logistic_regression(enc_X_train, Y_train, enc_X_test, Y_test, [100 50]);
% t_end = toc(t_start)

%model = logistic_regression(enc_X_train, Y_train, enc_X_test, Y_test, [100 75]);
% model = logistic_regression(X_train, Y_train, X_test, Y_test, [100 75]);

% 
% fprintf('Prediction from the logistic regression model trainined on the RBM\n')
% [label, score] = predict(model, enc_rec_image_h')
% 
% fprintf('Prediction from the logistic regression model trainined on the raw images\n')
% model = logistic_regression(X_train, Y_train, X_test, Y_test, [120 70]);
% [label, score] = predict(model, enc_rec_image')

% ------------ Train softmax layer ----------- %
% fprintf('\n--- Training softmax layer ---\n\n')
% t_start = tic;
% [theta, pred_Y_train, pred_Y_test] = softmax_layer(enc_X_train, Y_train, enc_X_test, Y_test, eps, MAX_EPOCHS_SOFTMAX, BATCH_SIZE);
% t_end = toc(t_start)

% ------ Show RBM Features ------- %
% plot_rbm_features(v, h, W)
% saveas(gcf, 'Results\rbm_features.png')



% ------------------ Utilities --------------------- %
% Draw a subplot with 'number of hidden units' features
% Basically for each hidden neuron we plot its weights as a figure
function plot_rbm_features(v, h, W)
    figure
    colormap gray
    min_v = min(min(W));
    max_v = max(max(W));
    size_subplot = sqrt(size(h, 1)); % 10 per column, 10 rows.
    for j = 1:size(h, 1)
        feature = reshape(W(:, j), 28, 28);
        subplot(size_subplot, size_subplot, j);
        image(255 * (feature - min_v) / (max_v - min_v));
        xticks([]); yticks([]);
    end
    sgtitle('100 features of the RBM')
end


function [X_train, Y_train, X_test, Y_test] = read_mnist(X_train_filename, Y_train_filename, X_test_filename, Y_test_filename)
    oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
    filenameImagesTrain = 'mnist-dataset\train-images-idx3-ubyte.gz';
    filenameLabelsTrain = 'mnist-dataset\train-labels-idx1-ubyte.gz';
    filenameImagesTest  = 'mnist-dataset\t10k-images-idx3-ubyte.gz';
    filenameLabelsTest  = 'mnist-dataset\t10k-labels-idx1-ubyte.gz';

    X_train = extractdata(processImagesMNIST(filenameImagesTrain));
    X_train = reshape(X_train, 784, 60000)';

    Y_train = processLabelsMNIST(filenameLabelsTrain);

    X_test = extractdata(processImagesMNIST(filenameImagesTest));
    X_test = reshape(X_test, 784, 10000)';

    Y_test = processLabelsMNIST(filenameLabelsTest);
    path(oldpath);
end