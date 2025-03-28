% MATLAB Code
function [offspring] = updateFunc1123(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    norm_cons = abs(cons) / (c_max + eps);
    
    % 2. Compute adaptive weights
    w = 1 ./ (1 + exp(-5*(0.6*norm_fit + 0.4*norm_cons)));
    w = w(:);
    
    % 3. Identify elite and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 4. Identify elite group and constraint violators
    k = max(2, floor(NP*0.15));
    [~, sort_fit] = sort(popfits);
    elite_idx = sort_fit(1:k);
    violators = find(cons > 0);
    if isempty(violators)
        violators = sort_fit(end-k+1:end);
    end
    
    % 5. Compute directional vectors
    w_elite = w(elite_idx);
    d_elite = sum((popdecs(elite_idx,:) - x_best) .* w_elite, 1) / (sum(w_elite) + eps;
    
    w_viol = w(violators);
    d_cons = sum((x_worst - popdecs(violators,:)) .* (1-w_viol), 1) / (sum(1-w_viol) + eps);
    
    % 6. Generate diversity component
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    r4 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    
    % 7. Hybrid mutation
    F1 = 0.8;
    F2 = 0.4;
    F3 = 0.6;
    mutants = popdecs + F1.*w.*d_elite + F2.*(1-w).*d_cons + F3.*(0.5-w).*d_div;
    
    % 8. Dynamic crossover
    CR = 0.95 - 0.45 * tanh(4*w);
    mask = rand(NP,D) < repmat(CR,1,D);
    j_rand = randi(D,NP,1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    out_lb = offspring < lb;
    out_ub = offspring > ub;
    offspring(out_lb) = 2*lb(out_lb) - offspring(out_lb);
    offspring(out_ub) = 2*ub(out_ub) - offspring(out_ub);
    offspring = min(max(offspring, lb), ub);
end