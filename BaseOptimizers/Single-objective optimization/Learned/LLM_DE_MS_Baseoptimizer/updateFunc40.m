function [offspring] = updateFunc40(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Sort population by fitness and get elite indices
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, floor(0.2*NP));
    elite_idx = sorted_idx(1:elite_size);
    
    % Normalize constraint violations
    cons_norm = sqrt(sum(cons.^2, 2)) + 1e-12;
    cons_dir = bsxfun(@rdivide, cons, cons_norm);
    
    % Get fitness statistics for adaptive scaling
    f_max = max(popfits);
    f_min = min(popfits);
    f_avg = mean(popfits);
    
    for i = 1:NP
        % Select elite base vector
        x_elite = popdecs(elite_idx(randi(elite_size)), :);
        
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        r = candidates(randperm(length(candidates), 2));
        x_r1 = popdecs(r(1), :);
        x_r2 = popdecs(r(2), :);
        
        % Calculate adaptive scaling factor
        if f_max ~= f_min
            F = 0.5 * (1 + (f_avg - popfits(i)) / (f_max - f_min));
        else
            F = 0.5;
        end
        
        % Calculate difference vector magnitude
        diff_mag = norm(x_r1 - x_r2);
        
        % Generate mutant vector
        offspring(i,:) = x_elite + F*(x_r1 - x_r2) + ...
                        0.5*cons_dir(i,:)*diff_mag;
        
        % Apply boundary check
        lb = min(popdecs);
        ub = max(popdecs);
        offspring(i,:) = min(max(offspring(i,:), lb), ub);
    end
end