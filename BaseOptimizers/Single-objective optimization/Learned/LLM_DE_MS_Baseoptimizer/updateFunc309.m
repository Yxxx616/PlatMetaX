% MATLAB Code
function [offspring] = updateFunc309(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate constraint violations
    abs_cons = max(0, cons);
    
    % Identify key individuals
    feasible_mask = cons <= 0;
    
    % Elite individual (best fitness among feasible)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx),:);
    else
        [~, elite_idx] = min(popfits);
        elite = popdecs(elite_idx,:);
    end
    
    % Best feasible individual (same as elite in this case)
    best_feas = elite;
    
    % Individual with lowest constraint violation
    [~, lowcons_idx] = min(abs_cons);
    lowcons = popdecs(lowcons_idx,:);
    
    % Generate unique random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Calculate direction vectors
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_cons = repmat(lowcons, NP, 1) - popdecs;
    
    % Normalize fitness and constraints for weights
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(abs_cons);
    c_max = max(abs_cons);
    norm_cons = (abs_cons - c_min) / (c_max - c_min + eps);
    
    % Adaptive weights
    sigma = 0.2;
    w_elite = exp(-norm_fits/sigma);
    w_feas = exp(-norm_cons/sigma);
    w_rand = 1 - exp(-norm_fits/sigma);
    w_cons = 1 - exp(-norm_cons/sigma);
    
    % Normalize weights
    w_total = w_elite + w_feas + w_rand + w_cons + eps;
    w_elite = w_elite ./ w_total;
    w_feas = w_feas ./ w_total;
    w_rand = w_rand ./ w_total;
    w_cons = w_cons ./ w_total;
    
    % Adaptive scaling factor based on rank
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.5 * (1 + tanh(2 * (0.5 - ranks/NP)));
    
    % Vectorized mutation
    offspring = popdecs + repmat(F,1,D) .* ...
               (repmat(w_elite,1,D).*d_elite + ...
                repmat(w_feas,1,D).*d_feas + ...
                repmat(w_rand,1,D).*d_rand + ...
                repmat(w_cons,1,D).*d_cons);
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (2*lb - offspring).*out_low + ...
               (2*ub - offspring).*out_high;
    
    % Final boundary enforcement with small perturbation
    perturb = 0.01*(ub-lb).*randn(NP,D).*repmat(1-ranks/NP,1,D);
    offspring = max(min(offspring + perturb, ub), lb);
end