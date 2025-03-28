% MATLAB Code
function [offspring] = updateFunc525(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    abs_cons = abs(cons);
    c_max = max(abs_cons);
    norm_con = abs_cons / (c_max + eps);
    
    % Weight calculation (cubic emphasis on fitness)
    w = (1 - norm_fit).^3 ./ (1 + norm_con.^2);
    
    % Elite selection with feasibility priority
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    opp_dir = repmat((lb + ub)/2, NP, 1) - popdecs;
    
    % Random directions
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    rand_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive scaling factors
    F_elite = 0.8 * w + 0.2;
    F_opp = 0.5 * (1 - w).^0.3;
    F_rand = 0.3 * (1 - w);
    
    % Mutation with multiple guidance
    offspring = popdecs + repmat(F_elite, 1, D) .* elite_dir + ...
                repmat(F_opp, 1, D) .* opp_dir + ...
                repmat(F_rand, 1, D) .* rand_dir;
    
    % Boundary handling with modular arithmetic
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    range = ub_rep - lb_rep;
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflected = lb_rep + mod(abs(offspring - lb_rep), range) .* (below | above);
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Final projection to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end