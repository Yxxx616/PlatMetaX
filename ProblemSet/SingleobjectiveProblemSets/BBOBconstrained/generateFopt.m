function offset = generateFopt(fNum, instance)
    switch fNum
        case 4
            rseed = 3;
        case 18
            rseed = 17;
        case {101, 102, 103, 107, 108, 109}
            rseed = 1;
        case {104, 105, 106, 110, 111, 112}
            rseed = 8;
        case {113, 114, 115}
            rseed = 7;
        case {116, 117, 118}
            rseed = 10;
        case {119, 120, 121}
            rseed = 14;
        case {122, 123, 124}
            rseed = 17;
        case {125, 126, 127}
            rseed = 19;
        case {128, 129, 130}
            rseed = 21;
        otherwise
            rseed = fNum;
    end

    % Compute instance-specific random seed
    rrseed = rseed + 10000 * instance;

    % Generate Gaussian random numbers
    rng(rrseed); % Set random seed
    gval = normrnd(0, 1); % Generate Gaussian random number with mean 0 and std 1
    rng(rrseed + 1); % Set random seed for the second random number
    gval2 = normrnd(0, 1); % Generate another Gaussian random number

    % Compute the offset
    offset = 100 * 100 * gval / gval2;
    offset = round(offset * 100) / 100; % Round to 2 decimal places
    offset = max(-1000, min(1000, offset)); % Clamp to [-1000, 1000]
end