function feasibleDirection = setFeasibleDirection(feasibleDirection, xopt, dimension, rseed)
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
    seed_offset = 412;  
    feas_shrink = 0.75; 
    feas_bound = 5.0;   

    maxabs = 0;
    maxrel = 0;
    
    
    for i = 1:dimension
        maxabs = max(maxabs, abs(xopt(i)));
        maxrel = max(maxrel, feasibleDirection(i) / (feas_bound - xopt(i)));
        maxrel = max(maxrel, feasibleDirection(i) / (-feas_bound - xopt(i)));
    end
    
   
    if maxabs > 4.01
        warning('feasible_direction_set_length: a component of fabs(xopt) was greater than 4.01');
    end

    if maxabs > 5.0
        error('feasible_direction_set_length: a component of fabs(xopt) was greater than 5.0');
    end
    
    rng(rseed + seed_offset);
    r = rand(1);
    
    feasibleDirection = feasibleDirection * (feas_shrink + r * (1 - feas_shrink)) / maxrel;
end

