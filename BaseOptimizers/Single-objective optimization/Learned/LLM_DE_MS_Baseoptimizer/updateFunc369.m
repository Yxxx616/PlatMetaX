% MATLAB Code
function [offspring] = updateFunc369(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate feasibility ratio
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask) / NP;
    
    % Normalize constraint violations (0 to 1)
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Normalize fitness values (0 to 1)
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Combined adaptive weights
    w = rho * norm_fits + (1 - rho) * (1 - norm_cons);
    
    % Select elite (best fitness) and leader (best feasible or least infeasible)
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx, :);
    
    if any(feasible_mask)
        [~, leader_idx] = min(popfits(feasible_mask));
        leader = popdecs(feasible_mask, :);
        leader = leader(leader_idx, :);
    else
        [~, leader_idx] = min(cons);
        leader = popdecs(leader_idx, :);
    end
    
    % Compute mutation vectors with adaptive scaling
    F1 = 0.7 * w;
    F2 = 0.3 * (1 - w);
    diff_leader = leader - popdecs;
    diff_elite = elite - popdecs;
    
    % Generate offspring with adaptive perturbation
    sigma = 0.25 * (1 - w);
    offspring = popdecs + F1 .* diff_leader + F2 .* diff_elite + sigma .* randn(NP, D);
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring .* (~out_low & ~out_high) + ...
        (2 * lb - offspring) .* out_low + ...
        (2 * ub - offspring) .* out_high;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
end