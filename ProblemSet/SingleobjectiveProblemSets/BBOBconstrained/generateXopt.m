function xopt = generateXopt(seed, DIM)
    % bbob2009_compute_xopt: Compute the optimal solution vector xopt
    % 
    % Input:
    %   seed - Random seed for reproducibility
    %   DIM - Dimension of the vector
    %
    % Output:
    %   xopt - The computed optimal solution vector

    % Initialize xopt with uniform random numbers in [0, 1)
    rng(seed); % Set the random seed for reproducibility
    xopt = rand(1, DIM);

    % Process each element of xopt
    for i = 1:DIM
        xopt(i) = 8 * floor(1e4 * xopt(i)) / 1e4 - 4;
        if xopt(i) == 0.0
            xopt(i) = -1e-5;
        end
    end
end