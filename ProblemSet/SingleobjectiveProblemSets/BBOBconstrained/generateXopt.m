function xopt = generateXopt(seed, DIM)
% bbob2009_compute_xopt: Compute the optimal solution vector xopt
% 
% Input:
%   seed - Random seed for reproducibility
%   DIM - Dimension of the vector
%
% Output:
%   xopt - The computed optimal solution vector

%------------------------------- Reference --------------------------------
% Paul Dufossé, Nikolaus Hansen, Dimo Brockhoff, Phillipe R. Sampaio, Asma 
% Atamna, and Anne Auger. Building scalable test problems for benchmarking 
% constrained optimizers. 2022. To be submitted to the SIAM Journal of 
% Optimization.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    rng(seed); 
    xopt = rand(1, DIM);

    for i = 1:DIM
        xopt(i) = 8 * floor(1e4 * xopt(i)) / 1e4 - 4;
        if xopt(i) == 0.0
            xopt(i) = -1e-5;
        end
    end
end