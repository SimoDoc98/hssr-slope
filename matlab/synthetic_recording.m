function [ppg,acc,t_beats] = synthetic_recording(duration,fs_ppg,fs_acc)
% synthetic_recording  Simulated wrist PPG corrupted by periodic motion.
%
%   [ppg,acc,t_beats] = synthetic_recording(duration,fs_ppg,fs_acc)
%   returns a PPG signal, the 3-axis acceleration and the times of the
%   simulated heart beats. Used by example_hssr_slope.m.

rng(0)

heart_rate = 75; % [bpm]
motion_frequency = 1.9; % [Hz]
motion_amplitude = 0.8;

% Beat times: mean interval modulated at 0.1 Hz and 0.25 Hz, plus jitter
t_beats = 0;
while t_beats(end) < duration + 2
    t = t_beats(end);
    ibi = 60/heart_rate * (1 + 0.04*sin(2*pi*0.1*t) + 0.03*sin(2*pi*0.25*t));
    t_beats(end+1) = t + ibi + 0.005*randn; %#ok<AGROW>
end

% Cardiac component: first and second harmonic of the beat phase
t = (0:duration*fs_ppg-1)'/fs_ppg;
phase = 2*pi*interp1(t_beats,0:length(t_beats)-1,t);
cardiac = cos(phase) + 0.35*cos(2*phase-0.8);

% Motion: periodic arm swing with slowly varying amplitude
swing = @(time) (1 + 0.3*sin(2*pi*0.03*time)) .* sin(2*pi*motion_frequency*time);

ppg = cardiac + motion_amplitude*swing(t) + 0.5*sin(2*pi*0.05*t) + 0.1*randn(size(t));

ta = (0:duration*fs_acc-1)'/fs_acc;
acc = [0.6*swing(ta), 0.3*swing(ta+0.05), 1 + 0.2*swing(ta)];
acc = acc + 0.02*randn(size(acc));

t_beats = t_beats(t_beats <= duration);

end
