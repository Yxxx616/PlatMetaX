% MATLAB Code
function [offspring] = updateFunc1191(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % 1. Calculate adaptive weights
    [~, fit_rank] = sort(popfits);
    w_fit = (NP - fit_rank + 1)' / sum(1:NP);
    
    [~, cons_rank] = sort(abs(cons));
    w_cons = (NP - cons_rank + 1)' / sum(1:NP);
    
    w = (w_fit .* w_cons) ./ (sum(w_fit .* w_cons) + eps);
    w = w / sum(w); % Normalize
    
    % 2. Select reference vectors
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 3. Generate random indices (distinct)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r3(i) == i || ...
              r1(i) == r2(i) || r1(i) == r3(i) || r2(i) == r3(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
            r3(i) = randi(NP);
        end
    end
    
    % 4. Adaptive scaling factors
    rho = sum(feasible_mask)/NP;
    F_e = 0.8 - 0.4*rho;  % More exploitation when feasible
    F_f = 0.5*rho;
    F_r = 0.5*(1-rho);
    
    % 5. Mutation components
    v_elite = popdecs + F_e * (x_best - popdecs);
    v_feas = popdecs + F_f * (x_feas - popdecs);
    v_rand = popdecs(r1,:) + F_r * (popdecs(r2,:) - popdecs(r3,:));
    v_opp = lb + ub - popdecs + 0.1*randn(NP,D);
    
    % 6. Weighted combination
    w_mat = [0.4*w, 0.3*w, 0.2*w, 0.1*w]; % Weight distribution
    mutants = w_mat(:,1).*v_elite + w_mat(:,2).*v_feas + ...
              w_mat(:,3).*v_rand + w_mat(:,4).*v_opp;
    
    % 7. Crossover with adaptive CR
    CR_base = 0.9;
    CR = CR_base * w + (1 - w) * 0.1;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final clamping to ensure within bounds
    offspring = max(min(offspring, ub), lb);
end