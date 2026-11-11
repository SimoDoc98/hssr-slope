function [idx_ppg_peaks,ibi,ppg_fpeak,flag_error,ppg_rec] = HSSR_SLOPE(ppg,acc,settings)
% HSSR_SLOPE  PPG waveform reconstruction and beat detection.
%
%   [idx_ppg_peaks,ibi,ppg_fpeak,flag_error,ppg_rec] = HSSR_SLOPE(ppg,acc,settings)
%
%   The PPG signal is decomposed with Singular Spectrum Analysis (SSA). The
%   components that fall in the first and second harmonic bands of the
%   cardiac frequency, and that are not correlated with the motion noise,
%   are summed to reconstruct the waveform. Beats are then detected with
%   the SLOPE algorithm.
%
%   Inputs
%     ppg       PPG signal (vector)
%     acc       3-axis acceleration (n-by-3 or 3-by-n), used as surrogate
%               of the motion noise. If empty, the noise surrogate is
%               obtained from the PPG signal itself (see note below).
%     settings  struct with fields
%       fs_ppg                 PPG sample frequency [Hz] (required)
%       fs_acc                 ACC sample frequency [Hz] (required with acc)
%       ppg_fpeak              characteristic frequency of the PPG [Hz];
%                              estimated from the signal if missing
%       rc                     Reconstruction Order, number of SSA
%                              components (default: 2*fs_ppg)
%       harmonics_tolerance    half-width of the harmonic bands [Hz]
%                              (default: 0.15)
%       correlation_tolerance  tolerance of the noise correlation
%                              criterion (default: 0.025)
%       ppg_band               pass band of the PPG filter [Hz]
%                              (default: [0.67 8])
%       delta_power_harmonics_tolerance
%                              the second harmonic is used only if its
%                              power is not lower than the first one by
%                              more than this value [dB] (default: -20)
%       visual_inspection      if true, the estimated ppg_fpeak is shown
%                              and can be confirmed or changed by the
%                              user (default: true)
%       ibi_outliers_method    "Grubbs" or "None" (default: "Grubbs")
%
%   Outputs
%     idx_ppg_peaks  sample indices of the detected beats
%     ibi            [time of the beat (s), inter-beat interval (ms),
%                    outlier flag]
%     ppg_fpeak      characteristic frequency of the PPG signal [Hz]
%     flag_error     1 if beat detection failed on the reconstructed
%                    signal and the pre-processed PPG was used instead
%     ppg_rec        reconstructed PPG signal
%
%   Note: the case without acceleration was not part of the validation
%   reported in the reference paper.
%
%   Copyright (c) 2025 Simone Costantini, Politecnico di Milano.
%   Distributed under the BSD 2-Clause License (see LICENSE).

%% Input checks

% Check on PPG vector size
if isempty(ppg)
    error("PPG is an empty array.")
elseif ~isvector(ppg)
    error("PPG is not a vector. Please, check its dimensions.")
end
ppg = ppg(:);

% Check on ACC array size
if isempty(acc)
    flag_acc_processing = false;
elseif ~ismatrix(acc) || ~any(size(acc) == 3)
    error("ACC is not a 3xn or a nx3 array. Please, check its dimensions.")
else
    if size(acc,1) == 3
        acc = acc';
    end
    flag_acc_processing = true;
end

% Check on PPG sample frequency
if ~isfield(settings,"fs_ppg") || ~(isnumeric(settings.fs_ppg) && isreal(settings.fs_ppg) && mod(settings.fs_ppg,1) == 0)
    error("Please, provide a valid sample frequency for PPG.")
end

% Check on ACC sample frequency
if flag_acc_processing
    if ~isfield(settings,"fs_acc") || ~(isnumeric(settings.fs_acc) && isreal(settings.fs_acc) && mod(settings.fs_acc,1) == 0)
        error("Please, provide a valid sample frequency for ACC.")
    end
end

%% Algorithm parameters setup

% Default values for HSSR-SLOPE parameters
default_settings.ppg_band = [0.67 8];
default_settings.harmonics_tolerance = 0.15;
default_settings.rc = settings.fs_ppg * 2;
default_settings.correlation_tolerance = 0.025;
default_settings.delta_power_harmonics_tolerance = -20;
default_settings.visual_inspection = true;
default_settings.ibi_outliers_method = "Grubbs";

% Missing settings are replaced by default values
names = fieldnames(default_settings);
for k = 1:numel(names)
    if ~isfield(settings,names{k})
        settings.(names{k}) = default_settings.(names{k});
    end
end

fs_ppg = settings.fs_ppg;
fcut_l = settings.ppg_band(1);
fcut_h = settings.ppg_band(2);
delta_band = settings.harmonics_tolerance;
N_RC = settings.rc;
tol_corr = settings.correlation_tolerance;
tol_delta_power_arm = settings.delta_power_harmonics_tolerance;

% Characteristic frequency of PPG signal
flag_fpeak_estimation = ~isfield(settings,"ppg_fpeak");

% PPG signal length
N_sig = size(ppg,1);

% Initialization of Error Flag
flag_error = 0;

%% PPG signal normalization

% PPG signal z-scoring
ppg = zscore(ppg);

% Building envelope on z-scored ppg
[env_up,env_low] = envelope(ppg,fs_ppg*2,"rms");

% Normalized PPG signal
ppg_norm = (ppg-env_low)./(env_up-env_low);

%% PPG signal filtering

% Band-pass filter
ppg_norm = bandpass(ppg_norm,[fcut_l,fcut_h]/(fs_ppg/2),"Steepness",0.85);

%% Noise surrogate estimation

if flag_acc_processing
    fs_acc = settings.fs_acc;

    % Resampling ACC to match the PPG sample frequency
    [p,q] = rat(fs_ppg/fs_acc);
    acc = resample(acc,p,q);

    % Building the modulus of the acceleration signal
    acc_mod = sqrt(sum(acc.^2,2));

    % ACC modulus z-score normalization
    acc_mod = zscore(acc_mod);

    % Matching ACC modulus and PPG signals lengths
    l_acc = length(acc_mod);

    if N_sig > l_acc % ACC modulus is extended with zero-padding
        acc_mod = [zeros(N_sig-l_acc,1);acc_mod];
    elseif N_sig < l_acc % ACC modulus is truncated
        acc_mod = acc_mod(1:N_sig);
    end

    noise = acc_mod;
else
    noise = wdenoise(ppg_norm,8, ...
        Wavelet='sym4', ...
        DenoisingMethod='bayes', ...
        ThresholdRule='Hard', ...
        NoiseEstimate='LevelDependent');
end

%% Estimation of the characteristic frequency of the PPG signal

if flag_fpeak_estimation

    % Spectrogram computation for PPG and noise
    [spectrogram_ppg,fr_ppg] = pspectrum(ppg_norm,fs_ppg,'spectrogram','Leakage',0.75,'OverlapPercent',0.5);
    spectrogram_noise = pspectrum(noise,fs_ppg,'spectrogram','Leakage',0.75,'OverlapPercent',0.5);

    % Power Spectral Density estimation of PPG and noise
    psd_ppg = mean(spectrogram_ppg,2);
    psd_noise = mean(spectrogram_noise,2);

    % Wiener filter H estimation
    H_wiener = rescale(psd_ppg,0,1) ./ (rescale(psd_ppg,0,1) + rescale(psd_noise,0,1));

    % Application of the Wiener filter to PPG in the frequency domain
    psd_ppg_clean = psd_ppg .* H_wiener;

    % Estimation of the PPG characteristic frequency
    [~,idx_fpeak] = max(psd_ppg_clean);
    ppg_fpeak = fr_ppg(idx_fpeak);

    fprintf("\nEstimation of PPG characteristic frequency: %0.2f Hz",ppg_fpeak)

    if settings.visual_inspection
        % Visual inspection
        figure()
        plot(fr_ppg,psd_ppg)
        hold on
        plot(fr_ppg,psd_noise)
        plot(fr_ppg,psd_ppg_clean)
        xline(ppg_fpeak)
        legend('PSD PPG raw','PSD Noise','PSD PPG clean','fpeak')
        xlim([0,3])
        xlabel("Frequency [Hz]")
        title("Please, confirm (spacebar) or modify (left click) the fpeak of PPG")
        grid on

        % Manual input from user
        [x_pos,~,button] = ginput(1);
        if button ~= 32
            ppg_fpeak = x_pos;
        end
        close

        fprintf("\nSelected PPG characteristic frequency: %0.2f Hz\n",ppg_fpeak)
    end
else
    ppg_fpeak = settings.ppg_fpeak;
end

%% Building the First and Second Harmonics Bands

harm1_band = [ppg_fpeak-delta_band,ppg_fpeak+delta_band];
harm2_band = [ppg_fpeak*2-delta_band,ppg_fpeak*2+delta_band];

%% Singular Spectrum Analysis

% Computing Reconstructed Components with SSA
RC = SSA(ppg_norm,N_RC);

% Reconstructed Components PSD estimation with Burg method
fr_RC = 0:0.01:fs_ppg/2;
psd_RC = zeros(length(fr_RC),N_RC);
for k = 1:N_RC
    psd_RC(:,k) = pburg(RC(:,k),4,fr_RC,fs_ppg);
end

%% Filtering Reconstructed Components - Noise correlation criterion

corr_noise = zeros(1,N_RC);
corr_ppg = zeros(1,N_RC);

for k = 1:N_RC
    corr_ppg(k) = corr(ppg,RC(:,k)); % Correlation between k-th RC and PPG
    corr_noise(k) = corr(noise,RC(:,k)); % Correlation between k-th RC and noise
end

delta_corr = corr_noise - corr_ppg;

RC(:,delta_corr>=tol_corr) = [];
psd_RC(:,delta_corr>=tol_corr) = [];

%% Out-Harmonics-Bands filtering and PPG Harmonics reconstruction

% Characteristic frequencies of Reconstructed Components
[~,idx] = max(psd_RC);
fr_RC_peaks = fr_RC(idx);

% First PPG harmonic reconstruction
inband = (harm1_band(1)<=fr_RC_peaks) & (fr_RC_peaks<=harm1_band(2));
ppg_harm1 = sum(RC(:,inband),2);

% Second PPG harmonic reconstruction
inband = (harm2_band(1)<=fr_RC_peaks) & (fr_RC_peaks<=harm2_band(2));
ppg_harm2 = sum(RC(:,inband),2);

% PPG harmonics power (in dB)
pwr_ppg_harm1 = 20*log10(sum(pwelch(ppg_harm1,[],[],[],fs_ppg)));
pwr_ppg_harm2 = 20*log10(sum(pwelch(ppg_harm2,[],[],[],fs_ppg)));

%% Reconstruction of PPG signal

if pwr_ppg_harm2 - pwr_ppg_harm1 < tol_delta_power_arm
    ppg_rec = ppg_harm1;
else
    ppg_rec = ppg_harm1 + ppg_harm2;
end

%% PPG fiducial points detection with SLOPE algorithm

% Time vector
v_time = (1:N_sig)/fs_ppg;

try
    [t_beats,idx_ppg_peaks] = PPG_FPDetection_Slope(ppg_rec,v_time,fs_ppg);
    if length(t_beats) < 3
        error("Too few beats detected.")
    end
catch
    % Beat detection on the pre-processed PPG signal
    warning("Beat detection failed on the reconstructed signal: the pre-processed PPG signal was used instead.")
    [t_beats,idx_ppg_peaks] = PPG_FPDetection_Slope(ppg_norm,v_time,fs_ppg);
    flag_error = 1;
end

%% Inter-beat intervals

t_beats = t_beats(:);
ibi = [t_beats(2:end),diff(t_beats)*1000];

switch settings.ibi_outliers_method
    case "Grubbs"
        ibi(:,3) = isoutlier(ibi(:,2),"grubbs");
    case "None"
        ibi(:,3) = 0;
    otherwise
        error("Unknown method for IBI outliers detection.")
end

end
