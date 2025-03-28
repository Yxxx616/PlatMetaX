% MATLAB Code
function [offspring] = updateFunc19(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations
    cons_abs = abs(cons);
    max_cons = max(cons_abs);
    cons_norm = cons_abs / (max_cons + 1e-12);
    
    % Sort population by fitness and get ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(1,NP);
    ranks(sorted_idx) = 1:NP;
    
    % Segment indices
    elite_num = max(1, round(0.2*NP));
    mid_num = round(0.6*NP);
    elite_idx = sorted_idx(1:elite_num);
    mid_idx = sorted_idx(elite_num+1:elite_num+mid_num);
    inf_idx = sorted_idx(elite_num+mid_num+1:end);
    
    % Generate all offspring in vectorized manner
    for i = 1:NP
        % Select base vectors
        x_elite = popdecs(elite_idx(randi(length(elite_idx))), :);
        x_mid = popdecs(mid_idx(randi(length(mid_idx))), :);
        x_inf = popdecs(inf_idx(randi(length(inf_idx))), :);
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        r = candidates(randperm(length(candidates), 2));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Adaptive scaling factors
        F_dir = 0.5 + 0.3 * (1 - ranks(i)/NP);
        F_cons = 0.3 * (1 + cons_norm(i));
        
        % Mutation
        offspring(i,:) = x_elite + F_dir.*(x_mid - x_inf) + F_cons.*(x_r1 - x_r2);
    end
    
    % Bound handling
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end