% MATLAB Code
function [offspring] = updateFunc1129(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c = max(0, cons);
    r_c = mean(c > 0);
    
    % 2. Adaptive weights and parameters
    norm_f = (popfits - f_mean) ./ f_std;
    w = exp(-norm_f.^2); % Fitness-based weights
    F = 0.5 * (1 + tanh(norm_f)); % Adaptive scaling factor
    CR = 0.9 - 0.5 * (c ./ (max(c) + eps)); % Constraint-aware CR
    
    % 3. Direction vectors
    % Elite direction
    elite_mask = w > 0.7;
    if any(elite_mask)
        elite_w = w(elite_mask);
        d_elite = sum((popdecs(elite_mask,:) - x_best) .* elite_w, 1) / (sum(elite_w) + eps;
    else
        d_elite = zeros(1,D);
    end
    
    % Diversity direction
    div_mask = w < 0.3;
    if any(div_mask)
        d_div = sum(x_worst - popdecs(div_mask,:), 1) / (sum(div_mask) + eps);
    else
        d_div = zeros(1,D);
    end
    
    % Constraint direction
    if any(c > 0)
        x_mean = mean(popdecs, 1);
        d_cons = sum((popdecs - x_mean) .* c, 1) / (sum(c) + eps);
    else
        d_cons = zeros(1,D);
    end
    
    % 4. Mutation with adaptive directions
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    alpha = 1 - r_c;
    beta = r_c;
    mutants = x_best + F.*(d_elite + alpha*d_div + beta*d_cons) + F.*(popdecs(r1,:) - popdecs(r2,:));
    
    % 5. Crossover with constraint awareness
    mask = rand(NP,D) < CR;
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with adaptive reflection
    out_of_bounds = (offspring < lb) | (offspring > ub);
    reflect = rand(NP,D) < 0.5;
    offspring(out_of_bounds & reflect) = min(max(2*lb(out_of_bounds & reflect) - offspring(out_of_bounds & reflect), lb), ub);
    offspring(out_of_bounds & ~reflect) = min(max(2*ub(out_of_bounds & ~reflect) - offspring(out_of_bounds & ~reflect), lb), ub);
    offspring = min(max(offspring, lb), ub);
end