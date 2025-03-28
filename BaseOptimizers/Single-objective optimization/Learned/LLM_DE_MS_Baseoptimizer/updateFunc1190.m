% MATLAB Code
function [offspring] = updateFunc1190(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-10;
    
    % 1. Calculate selection weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + eps;
    norm_fits = (popfits - mean_fit) ./ std_fit;
    std_cons = std(cons) + eps;
    norm_cons = cons ./ std_cons;
    w = 1./(1 + exp(5*norm_fits)) .* 1./(1 + exp(5*norm_cons));
    
    % 2. Select reference vectors
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        x_feas = mean(popdecs, 1);
    end
    
    % 3. Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)') | ...
           (r1 == r2) | (r1 == r3) | (r2 == r3);
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        r3(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == (1:NP)') | (r2 == (1:NP)') | (r3 == (1:NP)') | ...
               (r1 == r2) | (r1 == r3) | (r2 == r3);
    end
    
    % 4. Adaptive scaling factors
    rho = sum(feasible_mask)/NP;
    F_e = 0.7 + 0.2*rho;
    F_f = 0.5*rho;
    F_r = 0.3*(1-rho);
    
    % 5. Mutation components
    v_elite = popdecs + F_e * (x_best - popdecs);
    v_feas = popdecs + F_f * (x_feas - popdecs);
    v_rand = popdecs(r1,:) + F_r * (popdecs(r2,:) - popdecs(r3,:));
    
    % 6. Dynamic combination
    alpha = 0.5 * w;
    beta = 0.3 * (1 - abs(cons)/(max(abs(cons)) + eps);
    gamma = 0.2;
    
    mutants = alpha.*v_elite + beta.*v_feas + gamma.*v_rand;
    
    % 7. Crossover
    CR = 0.9*w + 0.1*(1 - abs(cons)/(max(abs(cons)) + eps);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    delta = 0.5;
    offspring(lb_mask) = lb(lb_mask) + delta*rand(sum(lb_mask(:)),1).*(popdecs(lb_mask)-lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - delta*rand(sum(ub_mask(:)),1).*(ub(ub_mask)-popdecs(ub_mask));
end