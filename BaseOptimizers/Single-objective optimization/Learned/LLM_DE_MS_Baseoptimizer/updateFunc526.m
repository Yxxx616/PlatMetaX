% MATLAB Code
function [offspring] = updateFunc526(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Enhanced normalization with sigmoid transformation
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = 1 ./ (1 + exp(-5*(popfits - f_min)/(f_max - f_min + eps)));
    
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = abs_cons / (c_max + eps);
    
    % Advanced weight calculation
    w = (1 - norm_fit).^2 ./ (1 + norm_con.^3);
    
    % Elite selection with dynamic feasibility consideration
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + norm_con * max(abs(popfits)));
        elite = popdecs(elite_idx,:);
    end
    
    % Opposition-based learning point
    opp_point = (lb + ub) - popdecs;
    
    % Random directions with guaranteed distinct indices
    r1 = randi(NP, NP, 1);
    r2 = mod(r1 + randi(NP-1, NP, 1), NP) + 1;
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive scaling factors with non-linear mapping
    F_elite = 0.7 * w + 0.3;
    F_opp = 0.4 * (1 - w).^0.5;
    F_rand = 0.2 * (1 - w);
    
    % Multi-directional mutation
    offspring = popdecs + repmat(F_elite, 1, D) .* (repmat(elite, NP, 1) - popdecs) + ...
                repmat(F_opp, 1, D) .* (opp_point - popdecs) + ...
                repmat(F_rand, 1, D) .* rand_dir;
    
    % Smart boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    range = ub_rep - lb_rep;
    
    % Reflection with modular arithmetic
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    reflected = lb_rep + mod(abs(offspring - lb_rep), range) .* below + ...
                ub_rep - mod(abs(offspring - ub_rep), range) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Final projection with small random perturbation near bounds
    near_bound = (offspring - lb_rep < 0.1*range) | (ub_rep - offspring < 0.1*range);
    offspring = offspring + near_bound .* (rand(NP,D)-0.5).*0.1.*range;
    offspring = max(min(offspring, ub_rep), lb_rep);
end