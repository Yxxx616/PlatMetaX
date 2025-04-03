function agent = DQN_NSGAII_MSDTLZ_Metaoptimizer(obsInfo, actInfo)
% DQN as meta-optimizer, being trained via RL.
% NSGAII as base-optimizer, with mutation strategy selection as
% opotimization object. Training set is DTLZ.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: A MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    agent = rlDDPGAgent(obsInfo,actInfo);
end
