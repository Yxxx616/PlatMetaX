% MATLAB Code
function [offspring] = updateFunc476(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility analysis
    feasible_mask = cons <= 0;
    alpha = sum(feasible_mask) / NP;
    
    % Elite selection with constraint handling
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalized constraints with smoothing
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive weights
    w_elite = 0.4*(1-alpha) + 0.1*randn(NP, 1);
    w_con = 0.3 * exp(-3 * norm_cons);
    w_rand = 0.3 * norm_cons.^2;
    
    % Expand weights to D dimensions
    w_elite = repmat(w_elite, 1, D);
    w_con = repmat(w_con, 1, D);
    w_rand = repmat(w_rand, 1, D);
    
    % Random indices selection
    r1 = randi(NP, NP, 1);
    r2 = arrayfun(@(x) setdiff(1:NP, x), 1:NP, 'UniformOutput', false);
    r2 = cellfun(@(x) x(randi(length(x))), r2)';
    
    % Random permutation for exploration
    rand1 = randperm(NP)';
    rand2 = randperm(NP)';
    
    % Mutation vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    rand_dir = popdecs(rand1,:) - popdecs(rand2,:);
    
    % Combined mutation
    offspring = popdecs + w_elite.*elite_dir + w_con.*diff_dir + w_rand.*rand_dir;
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability
    reflect_mask = rand(NP,D) < 0.7;
    
    % Boundary handling
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* (mask_low & reflect_mask) + ...
               (2*ub_rep - offspring) .* (mask_high & reflect_mask) + ...
               (lb_rep + (ub_rep-lb_rep).*rand(NP,D)) .* (mask_low | mask_high) .* ~reflect_mask;
    
    % Ensure numerical stability
    offspring = max(min(offspring, ub_rep), lb_rep);
end