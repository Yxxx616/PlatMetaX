function agent = DQN_DE_MS_Metaoptimizer(obsInfo, actInfo)
% Algorithm configuration-strategy contol
% DQN as meta-optimizer, being trained via RL.
% DE as base-optimizer, with mutation strategy selection as
% opotimization object.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 EvoSys_NUDT Group. You are free to use the PlatMetaX
% for research purposes. All publications which use this platform or MetaBBO
% code in the platform should acknowledge the use of "PlatMetaX" and 
% reference "Xu Yang, Rui Wang, Kaiwen Li, Wenhua Li, Tao Zhang and Fujun He. 
% PlatMetaX: An Integrated MATLAB platform for meta-black-box optimization.
% https://doi.org/10.48550/arXiv.2503.22722".
%--------------------------------------------------------------------------
    agent = rlDQNAgent(obsInfo,actInfo);
end
