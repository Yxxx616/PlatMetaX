% MATLAB Code
function [offspring] = updateFunc483(popdecs, popfits, cons)
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
    
    % Normalize constraints and fitness
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Adaptive weights
    w_elite = 0.5*alpha + 0.3*(1-alpha)*rand(NP, 1);
    w_con = 0.4 * exp(-2 * norm_cons);
    w_fit = 0.3 * (1 - norm_fits);
    sigma = 0.2 * (1-alpha);
    
    % Expand weights to D dimensions
    w_elite = repmat(w_elite, 1, D);
    w_con = repmat(w_con, 1, D);
    w_fit = repmat(w_fit, 1, D);
    
    % Random indices selection
    r1 = arrayfun(@(x) setdiff(1:NP, x), 1:NP, 'UniformOutput', false);
    r1 = cellfun(@(x) x(randi(length(x))), r1)';
    r2 = arrayfun(@(x) setdiff(1:NP, [x, r1(x)]), 1:NP, 'UniformOutput', false);
    r2 = cellfun(@(x) x(randi(length(x))), r2)';
    
    % Mutation components
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation
    offspring = popdecs + w_elite.*elite_dir + w_con.*diff_dir + ...
               sigma.*randn(NP,D).*w_fit;
    
    % Boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on feasibility
    reflect_prob = 0.5*alpha;
    reflect_mask = rand(NP,D) > reflect_prob;
    rand_mask = rand(NP,D) < reflect_prob;
    
    % Handle boundaries
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    % Reflection for some, random for others
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low .* reflect_mask + ...
               (2*ub_rep - offspring) .* mask_high .* reflect_mask + ...
               (lb_rep + (ub_rep-lb_rep).*rand(NP,D)) .* (mask_low | mask_high) .* rand_mask;
    
    % Final clipping for numerical stability
    offspring = max(min(offspring, ub_rep), lb_rep);
end