% MATLAB Code
function [offspring] = updateFunc1187(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Rank-based selection
    [~, rank_idx] = sort(popfits, 'descend');
    ranks = 1:NP;
    ranks(rank_idx) = ranks;
    rank_weights = 1./(ranks'.^0.5);
    
    % 2. Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    feasible_pop = popdecs(feasible_mask, :);
    if isempty(feasible_pop)
        feasible_pop = popdecs;
    end
    
    % 3. Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(mask_same)
        r1(mask_same) = randi(NP, sum(mask_same), 1);
        r2(mask_same) = randi(NP, sum(mask_same), 1);
        mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % 4. Select solutions based on rank weights
    elite_idx = randsample(NP, NP, true, rank_weights);
    x_elite = popdecs(elite_idx, :);
    
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    
    % 5. Enhanced feasibility weights
    abs_cons = abs(cons);
    max_cons = max(abs_cons) + 1e-10;
    w = 1./(1 + exp(-5*(1 - abs_cons./max_cons)));
    
    % 6. Adaptive scaling factors with rank influence
    F_elite = 0.7 - 0.4*rho;
    F_feas = 0.5*rho.^0.7;
    F_div = 0.3*(1-rho).^0.9;
    
    % 7. Three-component mutation with rank-based weights
    v_elite = popdecs + F_elite * (x_elite - popdecs);
    v_feas = popdecs + F_feas * (x_feas - popdecs) .* w;
    v_div = popdecs + F_div * (popdecs(r1,:) - popdecs(r2,:)) .* (1-w);
    
    % 8. Convex combination based on fitness ranks
    alpha = 0.4 + 0.3*(ranks'/NP);
    beta = 0.3 - 0.2*(ranks'/NP);
    gamma = 0.3 - 0.1*(ranks'/NP);
    
    mutants = alpha.*v_elite + beta.*v_feas + gamma.*v_div;
    
    % 9. Adaptive crossover with dynamic CR
    CR = 0.9 - 0.4*(1-w) + 0.1*(ranks'/NP);
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Smart boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + 0.3*rand(sum(lb_mask(:)),1).*(popdecs(lb_mask)-lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - 0.3*rand(sum(ub_mask(:)),1).*(ub(ub_mask)-popdecs(ub_mask));
end