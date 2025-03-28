% MATLAB Code
function [offspring] = updateFunc523(popdecs, popfits, cons)
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
    
    % Weight calculation with stronger emphasis on constraints
    w = (1 - norm_fit).^3 ./ (1 + norm_con.^2);
    
    % Elite selection with feasibility consideration
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Opposition-based directions
    opp_dir = repmat(lb + ub, NP, 1) - 2 * popdecs;
    
    % Elite directions
    elite_dir = repmat(elite, NP, 1) - popdecs;
    
    % Adaptive scaling factors
    F_elite = 0.8 * w.^1.5;
    F_opp = 0.4 * (1 - w).^0.8;
    sigma = 0.2 * (ub(1)-lb(1)) * (1 - w);
    
    % Mutation with elite and opposition guidance
    offspring = popdecs + repmat(F_elite, 1, D) .* elite_dir + ...
                repmat(F_opp, 1, D) .* opp_dir + ...
                repmat(sigma, 1, D) .* randn(NP, D);
    
    % Boundary handling with reflection and memory
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    reflected = (lb_rep + repmat(w, 1, D) .* (popdecs - lb_rep)) .* below + ...
                (ub_rep - repmat(w, 1, D) .* (ub_rep - popdecs)) .* above;
    
    offspring = offspring .* (~below & ~above) + reflected;
    
    % Controlled random diversity injection
    rand_mask = rand(NP, D) < 0.01 * repmat(1-w, 1, D);
    rand_vals = lb_rep + rand(NP, D) .* (ub_rep - lb_rep);
    offspring = offspring .* ~rand_mask + rand_vals .* rand_mask;
    
    % Final projection to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end