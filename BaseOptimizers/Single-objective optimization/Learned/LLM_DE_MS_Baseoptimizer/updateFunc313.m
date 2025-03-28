% MATLAB Code
function [offspring] = updateFunc313(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate constraint violations
    abs_cons = max(0, cons);
    
    % Identify key individuals
    feasible_mask = cons <= 0;
    
    % Elite individual (best feasible or least violating)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx),:);
    else
        [~, mincons_idx] = min(abs_cons);
        elite = popdecs(mincons_idx,:);
    end
    
    % Best feasible individual (if exists)
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask(best_feas_idx),:);
    else
        best_feas = elite;
    end
    
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
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    
    % Normalize fitness and constraints for weights
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs_cons);
    norm_cons = abs_cons / (c_max + eps);
    
    % Adaptive weights
    sigma_f = 0.2;
    sigma_c = 0.2;
    w_elite = exp(-norm_fits/sigma_f);
    w_feas = exp(-norm_cons/sigma_c);
    w_rand = ones(NP, 1);
    
    % Normalize weights
    w_total = w_elite + w_feas + w_rand + eps;
    w_elite = w_elite ./ w_total;
    w_feas = w_feas ./ w_total;
    w_rand = w_rand ./ w_total;
    
    % Rank-based scaling factor
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.5 + 0.5 * (ranks/NP);
    
    % Vectorized mutation
    offspring = popdecs + repmat(F,1,D) .* ...
               (repmat(w_elite,1,D).*d_elite + ...
                repmat(w_feas,1,D).*d_feas + ...
                repmat(w_rand,1,D).*d_rand);
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (2*lb - offspring).*out_low + ...
               (2*ub - offspring).*out_high;
    
    % Add adaptive perturbation based on rank
    perturb = 0.15*(ub-lb).*randn(NP,D).*repmat(1-ranks/NP,1,D);
    offspring = max(min(offspring + perturb, ub), lb);
    
    % Ensure diversity for top solutions
    explore_mask = ranks < max(3, 0.1*NP);
    if any(explore_mask)
        offspring(explore_mask,:) = offspring(explore_mask,:) + ...
            0.25*(ub-lb).*randn(sum(explore_mask),D);
        offspring = max(min(offspring, ub), lb);
    end
end