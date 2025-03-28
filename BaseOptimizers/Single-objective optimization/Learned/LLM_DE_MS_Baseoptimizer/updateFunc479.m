% MATLAB Code
function [offspring] = updateFunc479(popdecs, popfits, cons)
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
    
    % Best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    best = popdecs(best_idx, :);
    worst = popdecs(worst_idx, :);
    
    % Normalized constraints and fitness
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Adaptive weights
    w_elite = 0.4*alpha + 0.1*rand(NP, 1);
    w_con = 0.3 * exp(-3 * norm_cons);
    w_fit = 0.2 * (1 - norm_fits);
    sigma = 0.1 * (1-alpha);
    
    % Expand weights to D dimensions
    w_elite = repmat(w_elite, 1, D);
    w_con = repmat(w_con, 1, D);
    w_fit = repmat(w_fit, 1, D);
    
    % Random indices selection
    r1 = randi(NP, NP, 1);
    r2 = arrayfun(@(x) setdiff(1:NP, x), 1:NP, 'UniformOutput', false);
    r2 = cellfun(@(x) x(randi(length(x))), r2)';
    
    % Mutation vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    fit_dir = repmat(best, NP, 1) - repmat(worst, NP, 1);
    
    % Constraint-aware scaling
    con_scale = repmat(1 - norm_cons, 1, D);
    diff_dir = diff_dir .* con_scale;
    
    % Fitness-based scaling
    fit_scale = repmat(norm_fits, 1, D);
    fit_dir = fit_dir .* fit_scale;
    
    % Combined mutation
    offspring = popdecs + w_elite.*elite_dir + w_con.*diff_dir + ...
               w_fit.*fit_dir + sigma.*randn(NP,D);
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on feasibility
    reflect_prob = 0.6 + 0.3*alpha;
    reflect_mask = rand(NP,D) < reflect_prob;
    
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