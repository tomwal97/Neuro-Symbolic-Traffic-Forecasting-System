% ==============================================================================
% File: traffic_rules.pl
% Description: Symbolic Deterministic Rule Engine for Traffic State Inference
% ==============================================================================

% Dynamic predicates asserted at runtime by Python
:- dynamic predicted_speed/3.  % predicted_speed(NodeID, Step, Speed_mph)
:- dynamic baseline_speed/2.   % baseline_speed(NodeID, Speed_mph)
:- dynamic speed_limit/2.      % speed_limit(NodeID, Limit_mph)
:- dynamic connected/2.        % Graph connectivity edges from adjacency matrix A

% Default speed limit rule (65 mph default for PeMS highway corridor if not explicitly specified)
get_speed_limit(Node, Limit) :-
    speed_limit(Node, Limit), !.
get_speed_limit(_Node, 65.0).

% ------------------------------------------------------------------------------
% 1. Congestion & Level of Service (LOS) Rules
% ------------------------------------------------------------------------------

% Severe Congestion: Speed drops below 30.0 mph
is_severely_congested(Node, Step, Speed) :-
    predicted_speed(Node, Step, Speed),
    Speed < 40.0.

% Moderate Congestion: Speed between 30.0 mph and 45.0 mph
is_moderately_congested(Node, Step, Speed) :-
    predicted_speed(Node, Step, Speed),
    Speed >= 40.0,
    Speed < 55.0.

% Trigger a warning output when moderate congestion is detected
warning_congestion(Node, Step, Speed) :-
    is_moderately_congested(Node, Step, Speed).

% Free Flow Traffic: Speed strictly above 55.0 mph
is_free_flow(Node, Step, Speed) :-
    predicted_speed(Node, Step, Speed),
    Speed >= 55.0.



% ------------------------------------------------------------------------------
% 2. Temporal Anomaly & Shockwave Propagation Rules
% ------------------------------------------------------------------------------

% Fixed Sudden Bottleneck: Bind Node and StepNext via predicted_speed first
sudden_bottleneck(Node, StepNext, Drop) :-
    predicted_speed(Node, StepNext, SpeedNext),
    StepNext > 1,
    StepPrev is StepNext - 1,
    predicted_speed(Node, StepPrev, SpeedPrev),
    Drop is SpeedPrev - SpeedNext,
    Drop > 15.0.

% ------------------------------------------------------------------------------
% 3. Safety Guardrails & Enforcement Rules
% ------------------------------------------------------------------------------

% Fixed Unrealistic Speed Jump: Bind Node and StepNext via predicted_speed first
unrealistic_speed_jump(Node, StepNext, Increase) :-
    predicted_speed(Node, StepNext, SpeedNext),
    StepNext > 1,
    StepPrev is StepNext - 1,
    predicted_speed(Node, StepPrev, SpeedPrev),
    Increase is SpeedNext - SpeedPrev,
    Increase > 25.0.

% Speed Limit Enforcement Violation
is_speeding(Node, Step, Speed, Limit) :-
    predicted_speed(Node, Step, Speed),
    get_speed_limit(Node, Limit),
    Speed > Limit.

% ------------------------------------------------------------------------------
% 4. Advanced Spatial Propagation & Spillback Rules
% ------------------------------------------------------------------------------

% Spatial Spillback: If an upstream node is severely congested at step T,
% a connected downstream node experiences secondary congestion at step T+1.
spatial_spillback(Upstream, Downstream, StepNext) :-
    connected(Upstream, Downstream),
    is_severely_congested(Upstream, Step, _),
    StepNext is Step + 1,
    is_severely_congested(Downstream, StepNext, SpeedDown),
    SpeedDown < 35.0.