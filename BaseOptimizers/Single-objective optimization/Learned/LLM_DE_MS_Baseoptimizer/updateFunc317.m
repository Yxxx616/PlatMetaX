% MATLAB Code
function [offspring] = updateFunc317(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Constraint handling
    abs_cons = max(0, cons);
    feasible_mask = cons <= 0;
    
    % Elite selection (best feasible or least infeasible)
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
    
    % Calculate distance to elite
    d_elite = mean(abs(popdecs - repmat(elite, NP, 1)) ./ (ub - lb);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(abs_cons);
    norm_cons = abs_cons / (c_max + eps);
    
    % Adaptive weights
    sigma_f = 0.2;
    sigma_c = 0.2;
    sigma_d = 0.1;
    
    w_elite = exp(-norm_fits/sigma_f);
    w_feas = exp(-norm_cons/sigma_c);
    w_dist = exp(-d_elite'/sigma_d);
    w_total = w_elite + w_feas + w_dist + eps;
    
    w_elite = w_elite ./ w_total;
    w_feas = w_feas ./ w_total;
    w_rand = 1 - w_elite - w_feas;
    
    % Rank-based scaling factor
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_idx) = (1:NP)';
    F = 0.3 + 0.5 * (1 - ranks/NP).^2;
    
    % Direction vectors
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation
    offspring = popdecs + repmat(F,1,D) .* ...
               (repmat(w_elite,1,D).*d_elite + ...
                repmat(w_feas,1,D).*d_feas + ...
                repmat(w_rand,1,D).*d_rand);
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    range = ub - lb;
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + abs(offspring - lb)) .* out_low + ...
               (ub - abs(offspring - ub)) .* out_high;
    
    % Elite-guided perturbation for top 20% solutions
    top_mask = ranks <= max(3, 0.2*NP);
    if any(top_mask)
        perturb = 0.1*(ub-lb).*(rand(sum(top_mask),D)-0.5);
        offspring(top_mask,:) = offspring(top_mask,:) + perturb;
    end
    
    % Ensure boundaries
    offspring = max(min(offspring, ub), lb);
    
    % Diversity enhancement for worst solutions
    worst_mask = ranks > 0.8*NP;
    if any(worst_mask)
        offspring(worst_mask,:) = lb + (ub-lb).*rand(sum(worst_mask),D);
    end
end