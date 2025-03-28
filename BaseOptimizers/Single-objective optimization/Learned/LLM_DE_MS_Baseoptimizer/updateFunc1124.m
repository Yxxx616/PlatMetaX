% MATLAB Code
function [offspring] = updateFunc1124(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 2. Compute adaptive weights based on fitness and constraints
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    w_fit = 1 ./ (1 + exp(5*(popfits - f_mean)/f_std));
    w_cons = 1 ./ (1 + exp(-5*abs(cons)/c_max));
    w = 0.7*w_fit + 0.3*w_cons;
    w = w(:);
    
    % 3. Identify elite group (top 20%) and constraint violators
    k_elite = max(2, floor(NP*0.2));
    [~, sort_idx] = sort(popfits);
    elite_idx = sort_idx(1:k_elite);
    violators = find(cons > 0);
    if isempty(violators)
        violators = sort_idx(end-k_elite+1:end);
    end
    
    % 4. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps);
    
    w_viol = w(violators);
    d_cons = sum((x_worst - popdecs(violators,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 5. Generate differential component
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_diff = popdecs(r1,:) - popdecs(r2,:);
    
    % 6. Hybrid mutation with adaptive factors
    F1 = 0.8;
    F2 = 0.4;
    F3 = 0.6;
    mutants = popdecs + F1.*w.*d_elite + F2.*(1-w).*d_cons + F3.*d_diff;
    
    % 7. Constraint-aware crossover
    CR = 0.9 * (1 - abs(cons)/c_max);
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with bounce-back
    out_lb = offspring < lb;
    out_ub = offspring > ub;
    offspring(out_lb) = lb(out_lb) + rand(sum(out_lb(:)),1).*(popdecs(out_lb) - lb(out_lb));
    offspring(out_ub) = ub(out_ub) - rand(sum(out_ub(:)),1).*(ub(out_ub) - popdecs(out_ub));
end