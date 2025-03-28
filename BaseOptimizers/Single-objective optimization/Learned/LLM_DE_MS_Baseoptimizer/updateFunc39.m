function [offspring] = updateFunc39(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness and get ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    
    % Determine elite group (top 20%)
    elite_size = max(1, floor(0.2*NP));
    elite_idx = sorted_idx(1:elite_size);
    
    % Normalize constraint violations with epsilon for numerical stability
    cons_norm = sqrt(sum(cons.^2, 2)) + 1e-12;
    cons_dir = bsxfun(@rdivide, cons, cons_norm);
    
    % Get dynamic bounds for opposition-based learning
    lb = min(popdecs);
    ub = max(popdecs);
    
    for i = 1:NP
        % Select elite base vector
        x_elite = popdecs(elite_idx(randi(elite_size)), :);
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        r = candidates(randperm(length(candidates), 2));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Calculate adaptive parameters
        F = 0.4 + 0.4 * (1 - ranks(i)/NP);
        p_opposition = 0.3 * (1 - ranks(i)/NP);
        
        % Calculate difference vector norm
        diff_norm = norm(x_r1 - x_r2);
        
        % Generate mutant vector
        offspring(i,:) = x_elite + F*(x_r1 - x_r2) + ...
                        0.5*cons_dir(i,:)*diff_norm;
        
        % Apply opposition-based learning with adaptive probability
        if rand() < p_opposition
            offspring(i,:) = lb + ub - offspring(i,:);
        end
    end
end