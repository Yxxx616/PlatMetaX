% MATLAB Code
function [offspring] = updateFunc1188(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Calculate selection weights
    mean_fit = mean(popfits);
    std_fit = std(popfits) + 1e-10;
    std_cons = std(cons) + 1e-10;
    norm_fits = (popfits - mean_fit)/std_fit;
    norm_cons = cons/std_cons;
    w = 1./(1 + exp(-5*norm_fits)) .* 1./(1 + exp(5*norm_cons));
    
    % 2. Select best and feasible solutions
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = popdecs(feasible_mask, :);
    else
        x_feas = popdecs;
    end
    
    % 3. Generate random indices for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(mask_same)
        r1(mask_same) = randi(NP, sum(mask_same), 1);
        r2(mask_same) = randi(NP, sum(mask_same), 1);
        mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % 4. Calculate adaptive scaling factors
    rho = sum(feasible_mask)/NP;
    F_e = 0.6 + 0.2*rho;
    F_f = 0.4*rho;
    F_d = 0.3*(1-rho);
    
    % 5. Generate mutation components
    v_elite = popdecs + F_e * (x_best - popdecs);
    feas_idx = randi(size(x_feas,1), NP, 1);
    v_feas = popdecs + F_f * (x_feas(feas_idx,:) - popdecs);
    v_div = popdecs + F_d * (popdecs(r1,:) - popdecs(r2,:));
    
    % 6. Dynamic combination weights
    [~, ranks] = sort(popfits);
    alpha = 0.5 * (1 + cos(pi*ranks/NP));
    beta = 0.3 * (1 - abs(cons)/max(abs(cons)+1e-10));
    gamma = 0.2 * (1 - w);
    
    % 7. Combine components
    mutants = alpha.*v_elite + beta.*v_feas + gamma.*v_div;
    
    % 8. Adaptive crossover
    CR = 0.9*w + 0.1*(1 - abs(cons)/max(abs(cons)+1e-10);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + 0.5*rand(sum(lb_mask(:)),1).*(popdecs(lb_mask)-lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - 0.5*rand(sum(ub_mask(:)),1).*(ub(ub_mask)-popdecs(ub_mask));
end