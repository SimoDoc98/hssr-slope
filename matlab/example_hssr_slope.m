% Example: HSSR-SLOPE on a simulated wrist recording

clear
close all
clc

%% Simulated PPG and acceleration signals

fs_ppg = 64;
fs_acc = 32;

[ppg,acc,t_true] = synthetic_recording(120,fs_ppg,fs_acc);

%% HSSR-SLOPE

settings.fs_ppg = fs_ppg;
settings.fs_acc = fs_acc;

% Optional parameters (default values are used if not specified)
% settings.rc = 128;                      % Reconstruction Order
% settings.harmonics_tolerance = 0.15;    % Harmonics Tolerance [Hz]
% settings.correlation_tolerance = 0.025; % Correlation Tolerance
% settings.visual_inspection = false;     % skip the check of the fpeak

% A figure shows the spectra of the PPG signal and of the noise surrogate:
% press the spacebar to confirm the estimated characteristic frequency, or
% click on the correct peak.
[idx_peaks,ibi,ppg_fpeak,flag_error,ppg_rec] = HSSR_SLOPE(ppg,acc,settings);

%% Results

ibi_true = diff(t_true)*1000;

fprintf("\nCharacteristic frequency: %0.2f Hz",ppg_fpeak)
fprintf("\nDetected beats: %d (simulated: %d)",length(idx_peaks),length(t_true))
fprintf("\nMean IBI: %0.1f ms (simulated: %0.1f ms)",mean(ibi(:,2)),mean(ibi_true))
fprintf("\nSDNN: %0.1f ms (simulated: %0.1f ms)\n",std(ibi(:,2)),std(ibi_true))

t = (1:length(ppg))/fs_ppg;

figure()

subplot(3,1,1)
plot(t,ppg)
xlim([20 40])
title("Raw PPG")

subplot(3,1,2)
plot(t,ppg_rec)
hold on
plot(t(idx_peaks),ppg_rec(idx_peaks),'o')
xlim([20 40])
title("Reconstructed PPG and detected beats")

subplot(3,1,3)
plot(t_true(2:end),ibi_true)
hold on
plot(ibi(:,1),ibi(:,2))
xlabel("Time [s]")
ylabel("IBI [ms]")
legend("Simulated","HSSR-SLOPE")
title("Inter-beat intervals")
