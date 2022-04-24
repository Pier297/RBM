
function [enc_X_train, enc_X_test, v, h, W, bias_v, bias_h, rec_image, image_noise] = RBM(X_train, X_test, n_hidden_units, k, eps, MAX_EPOCHS, BATCH_SIZE)
    [v, h, W, bias_v, bias_h] = create_rbm(size(X_train, 2), n_hidden_units);
    [v, h, W] = learn(X_train, v, h, W, bias_v, bias_h, MAX_EPOCHS, BATCH_SIZE, eps, k);
    
    % --- Encode the images ---
    enc_X_train = zeros(size(X_train, 1), n_hidden_units);
    for i = 1:size(X_train, 1)
        enc_X_train(i, :) = logistic(X_train(i, :) * W + bias_h')';
    end
    enc_X_test = zeros(size(X_test, 1), n_hidden_units);
    for i = 1:size(X_test, 1)
        enc_X_test(i, :) = logistic(X_test(i, :) * W + bias_h')';
    end
    
    % --- Test reconstruction ---
    sample_image = X_test(1, :);
    % This is a simple test, it projects the image into the hidden
    % representation and then it reconstructs the image.
%     demo_reconstruction(sample_image', h, W, bias_v, bias_h);
    
    % --- Test reconstruct noisy image ---
    rec_image = [];
    image_noise = [];
%     [rec_image, image_noise] = demo_denoise(sample_image', h, W, bias_v, bias_h);
    
%     demo_full_reconstruction(X_test(16,:)', h, W, bias_v, bias_h)
    
    % Create random image
    random_img = zeros(28, 28);
%     random_img = randn(28, 28);
%     min_v = min(min(random_img));
%     max_v = max(max(random_img));
%     random_img = (random_img - min_v) / (max_v - min_v);
    dream(reshape(random_img, 28*28, 1), h, W, bias_v, bias_h)
end

% ------------------------ Utilities ------------------------ %
function [v, h, W, bias_v, bias_h] = create_rbm(n_visible, n_hidden)
    h = zeros(n_hidden, 1);
    v = zeros(n_visible, 1);
    % Randomly initialized weights with a zero-mean Gaussian with sd=0.01
    sd = 0.01;
    W = sd * randn(n_visible, n_hidden);
    bias_v = zeros(n_visible, 1);
    bias_h = zeros(n_hidden, 1);
end

function dream(v, h, W, bias_v, bias_h)
    % Show original image
    figure
    colormap gray
    image(floor(255 * reshape(v, 28, 28)))
    title('Original Image')
    saveas(gcf, 'Results\Demo4\original.png')
    
    it = 1;
    energies = [energy(v, h, W, bias_v, bias_h)];
    k = 5;
    for s = 1:5
        h = logistic(v' * W + bias_h')' > rand(size(h, 1), 1);
        v = logistic(W * h + bias_v);
        it = it + 1;
        energies(end + 1) = energy(v, h, W, bias_v, bias_h);
        figure
        colormap gray
        image(floor(255 * reshape(v, 28, 28)))
        title('Reconstructed Image ' + string(it - 1))
        saveas(gcf, 'Results\Demo4\dream_' + string(it - 1) + '.png')
    end
    
    figure
    plot(0:it-1, energies);
    xlabel('state')
    ylabel('energy')
    title('Energy function')
    saveas(gcf, 'Results\Demo4\energy.png')
end

function [rec_image, noisy_image] = demo_denoise(v, h, W, bias_v, bias_h)
    img = v;
    h = logistic(v' * W + bias_h')' > rand(size(h, 1), 1);
    v = logistic(W * h + bias_v);
    % Show original image
    figure
    colormap gray
    image(floor(255 * reshape(img, 28, 28)))
    title('Original Image')
    saveas(gcf, 'Results\Demo2\original.png')
    
    v = v + randn(size(v, 1), 1);
    v = v / max(v);
    noisy_image = v;
    figure
    colormap gray
    image(floor(255 * reshape(v, 28, 28)))
    title('Noisy Image')
    saveas(gcf, 'Results\Demo2\noisy.png')
    
    p_j = logistic(v' * W + bias_h')';
    h = p_j > rand(size(h, 1), 1);
    
    % Show inner representation h
    figure
    colormap gray
    image(floor(255 * reshape(h, 10, 10)))
    title('Hidden represantion')
    saveas(gcf, 'Results\Demo2\hidden_representation.png')
    
    v = logistic(W * h + bias_v);
    rec_image = p_j;
    figure
    colormap gray
    image(floor(255 * reshape(v, 28, 28)))
    title('Reconstructed Image')
    saveas(gcf, 'Results\Demo2\reconstructed.png')
end

% Takes an image clamped on v and tries to reconstruct it from it's
% representation in h
function demo_reconstruction(v, h, W, bias_v, bias_h)
    img = v;
    h = logistic(v' * W + bias_h')' > rand(size(h, 1), 1);
    v = logistic(W * h + bias_v);
    % Show original image
    figure
    colormap gray
    image(floor(255 * reshape(img, 28, 28)))
    title('Original Image')
    saveas(gcf, 'Results\Demo1\original.png')
    % Show inner representation h
    figure
    colormap gray
    image(floor(255 * reshape(h, 10, 10)))
    title('Hidden represantion')
    saveas(gcf, 'Results\Demo1\hidden_representation.png')
    % Show reconstructed image
    figure
    colormap gray
    image(floor(255 * reshape(v, 28, 28)))
    title('Reconstructed Image')
    saveas(gcf, 'Results\Demo1\reconstruction.png')
end

function E = energy(v, h, W, bias_v, bias_h)
    E = -(v' * (W * h)) - (bias_v' * v) - (bias_h' * h);
end

function r = logistic(x)
    r = 1 ./ (1 + exp(-x));
end

function [W, bias_v, bias_h] = mini_batch(X, v, h, W, bias_v, bias_h, eps, k)
    delta_W = zeros(size(v, 1), size(h, 1));
    delta_bias_v = zeros(size(v, 1), 1);
    delta_bias_h = zeros(size(h, 1), 1);
    % For each data point of the mini-batch, apply CD training
    for i = 1:size(X, 1)
        v_0 = X(i, :)';
        h_0 = logistic(v_0' * W + bias_h')' > rand(size(h, 1), 1);
        % k step gibbs sampling
        h_k = h_0;
        for s = 1:k
            v_k = logistic(W * h_k + bias_v);
            p_j = logistic(v_k' * W + bias_h')';
            h_k = p_j > rand(size(h, 1), 1);
        end
        h_k = p_j;
        % When computing the gradient use p_j instead of h_j,
        % this reduces the sampling noise -> leads to faster training
        % -- [Hinton, A Practical Guide to Training Restricted Boltzmann Machines]
        delta_W = delta_W + (v_0 * h_0') - (v_k * h_k');
        delta_bias_v = delta_bias_v + (v_0 - v_k);
        delta_bias_h = delta_bias_h + (h_0 - h_k);
    end
    % Update W
    W = W + (eps / size(X, 1)) * delta_W;
    % Update biases
    bias_v = bias_v + (eps / size(X, 1)) * delta_bias_v;
    bias_h = bias_h + (eps / size(X, 1)) * delta_bias_h;
end

function [v, h, W] = learn(X, v, h, W, bias_v, bias_h, MAX_EPOCHS, BATCH_SIZE, eps, k)    
    for t = 1:MAX_EPOCHS
        t_start = tic;
        % Shuffle the training data
        perm = randperm(size(X, 1));
        X = X(perm, :);
        
        % Mini-Batch learning
        for i = 1:ceil(size(X, 1) / BATCH_SIZE)
            pre_index = (i - 1) * BATCH_SIZE + 1;
            batch_X = X(pre_index:min(end, i*BATCH_SIZE), :);
            [W, bias_v, bias_h] = mini_batch(batch_X, v, h, W, bias_v, bias_h, eps, k);
        end
        
        t_end = toc(t_start);
        % Print current total reconstruction error
        fprintf('Epoch %d | error = %d | time = %.2f seconds\n', t, total_reconstruction_error(X, v, h, W, bias_v, bias_h), t_end);
    end
end

function r = reconstruction_error(v, h, W, bias_v, bias_h)
    h = logistic(v' * W + bias_h')' > rand(size(h, 1), 1);
    p_v = logistic(W * h + bias_v);
    r = norm(p_v-v)^2;
end

function r = total_reconstruction_error(X, v, h, W, bias_v, bias_h)
    r = 0.0;
    for i = 1:size(X, 1)
        r = r + reconstruction_error(X(i, :)', h, W, bias_v, bias_h);
    end
end