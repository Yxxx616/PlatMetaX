% MATLAB Code
function [offspring] = updateFunc335(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    
    % Elite selection (best fitness)
    [~, elite_idx] = min(popfits);
    elite = popdecs(elite_idx,:);
    
    % Best feasible individual
    if any(feasible_mask)
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas = popdecs(feasible_mask(best_feas_idx),:);
    else
        [~, best_cons_idx] = min(abs_cons);
        best_feas = popdecs(best_cons_idx,:);
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
    
    % Adaptive parameters
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_std = std(abs_cons) + eps;
    
    % Adaptive weights
    alpha = 1./(1 + exp(-(popfits - f_mean)./f_std));
    beta = 1./(1 + exp(abs_cons./c_std));
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.3 + 0.5 * (1 - sqrt(ranks/NP));
    
    % Direction vectors
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with adaptive weights
    offspring = popdecs + repmat(F,1,D) .* ...
               (repmat(beta,1,D).*d_feas + ...
                repmat(1-beta,1,D).*(repmat(alpha,1,D).*d_elite + ...
                repmat(1-alpha,1,D).*d_rand));
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (2*lb - offspring).*out_low + ...
               (2*ub - offspring).*out_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end