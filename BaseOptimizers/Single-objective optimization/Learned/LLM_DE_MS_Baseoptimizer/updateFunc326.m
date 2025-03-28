% MATLAB Code
function [offspring] = updateFunc326(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    
    % Elite selection
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx),:);
    else
        [~, mincons_idx] = min(abs_cons);
        elite = popdecs(mincons_idx,:);
    end
    
    % Best feasible individual
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
    
    % Adaptive parameters
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_mean = mean(abs_cons);
    c_std = std(abs_cons) + eps;
    
    % Adaptive weights
    alpha = 0.5 * (1 + tanh((popfits - f_mean)./f_std));
    beta = 0.5 * (1 - tanh((abs_cons - c_mean)./c_std));
    
    % Rank-based scaling factors
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.3 + 0.5 * (1 - (ranks/NP).^0.3);
    
    % Direction vectors
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation
    offspring = popdecs + repmat(F,1,D) .* ...
               (repmat(beta,1,D).*d_feas + ...
                repmat(1-beta,1,D).*(repmat(alpha,1,D).*d_elite + ...
                                    repmat(1-alpha,1,D).*d_rand));
    
    % Boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    range = ub - lb;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + abs(mod(offspring - lb, range))) .* out_low + ...
               (ub - abs(mod(offspring - ub, range))) .* out_high;
    
    % Diversity enhancement for stagnant solutions
    stagnant_mask = ranks > 0.8*NP;
    if any(stagnant_mask)
        offspring(stagnant_mask,:) = lb + (ub-lb).*rand(sum(stagnant_mask),D);
    end
    
    % Final boundary check
    offspring = max(min(offspring, ub), lb);
end