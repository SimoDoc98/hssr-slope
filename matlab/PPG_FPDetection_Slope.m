function [t_peaks,idx_peaks] = PPG_FPDetection_Slope(BVP,t_BVP,fs_bvp)
% PPG_FPDetection_Slope  SLOPE algorithm for PPG beat detection.
%
%   [t_peaks,idx_peaks] = PPG_FPDetection_Slope(BVP,t_BVP,fs_bvp)
%
%   An adaptive threshold follows the PPG signal up to a local maximum,
%   which is marked as a beat. The threshold then decreases linearly, with
%   an adaptive slope, until it crosses the signal again after the
%   refractory period.
%
%   Inputs
%     BVP      PPG signal (vector)
%     t_BVP    time vector of the PPG signal [s]
%     fs_bvp   sample frequency [Hz]
%
%   Outputs
%     t_peaks    time of the detected beats [s]
%     idx_peaks  sample indices of the detected beats
%
%   Copyright (c) 2023 Simone Costantini, Politecnico di Milano.
%   Distributed under the BSD 2-Clause License (see LICENSE).

x = detrend(BVP);
t = t_BVP;
N = length(x);

t_peaks = [];
idx_peaks = [];

RP = 0.4; % Refractory period [s]

sr = 0.6; % Update rate of the slope
alpha = 0.8; % Scaling of the slope

slope = alpha*max(x);

% The search starts from the first trough of the signal
[~,min_idx] = findpeaks(-BVP,"MinPeakWidth",0.3,"MinPeakHeight",0);

idx_init = min_idx(1);

th = zeros(1,N);

th(idx_init) = x(idx_init);

i = idx_init;

first_cycle = false;

while i < N

    % The threshold follows the signal up to the next local maximum
    i2 = i + 1;

    while i2 < N
        if x(i2) > x(i2+1)
            j = i2;
            th(i2) = x(j);
            break
        else
            th(i2) = x(i2);
            i2 = i2 + 1;
        end
    end

    if first_cycle
        if t_peaks(end) == t(j)
            break
        else
            t_peaks = [t_peaks,t(j)];
            idx_peaks = [idx_peaks,j];
        end
    else
        t_peaks = [t_peaks,t(j)];
        idx_peaks = [idx_peaks,j];
        first_cycle = true;
    end

    slope = (1-sr)*slope + sr*alpha*x(j); % Slope update

    % Linear decrease of the threshold until it crosses the signal
    i3 = j + 1;

    th(i3) = th(j)-slope/fs_bvp;

    while i3 <= N-1

        if th(i3) < x(i3) && (i3 - j)/fs_bvp >= RP
            i = i3 + 1;
            th(i) = x(i);
            break
        else
            i3 = i3 + 1;
            th(i3) = th(i3-1) - slope/fs_bvp;
        end

    end

end

end
