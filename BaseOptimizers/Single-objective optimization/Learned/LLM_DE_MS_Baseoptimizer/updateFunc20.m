% MATLAB Code
function [offspring] = updateFunc20(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations
    cons_abs = abs(cons);
    max_cons = max(cons_abs);
    cons_norm = cons_abs / (max_cons + 1e-12);
    
    % Sort population by fitness and get ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    ranks = ranks / NP; % Normalize ranks to [0,1]
    
    % Elite selection (top 20%)
    elite_num = max(1, round(0.2*NP));
    elite_idx = sorted_idx(1:elite_num);
    
    % Generate all offspring in vectorized manner
    for i = 1:NP
        % Select elite base vector
        x_elite = popdecs(elite_idx(randi(length(elite_idx))), :);
        
        % Select four distinct random vectors (excluding current and elite)
        candidates = setdiff(1:NP, [i; elite_idx]);
        r = candidates(randperm(length(candidates), 4));
        x_r2 = popdecs(r(1), :);
        x_r3 = popdecs(r(2), :);
        x_r4 = popdecs(r(3), :);
        x_r5 = popdecs(r(4), :);
        
        % Adaptive scaling factors
        F1 = 0.5 * (1 - ranks(i)); % Rank-based scaling
        F2 = 0.3 * (1 + cons_norm(i)); % Constraint-based scaling
        
        % Mutation operation
        offspring(i,:) = x_elite + F1.*(x_r2 - x_r3) + F2.*(x_r4 - x_r5);
    end
    
    % Bound handling
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = min(max(offspring, lb), ub);
end