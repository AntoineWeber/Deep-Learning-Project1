close all

% Trial to plot
plot_trial = 74;

% X has shape [samples, nb_electrodes*time_samples]
X_train = importdata("data/sp1s_aa_train_1000Hz.txt");
X_test = importdata("data/sp1s_aa_test_1000Hz.txt");
X_train = X_train(:,2:end);     % removing labels
nb_time_samples = 500;
nb_electrodes = 28;
nb_trials_train = size(X_train, 1);
nb_trials_test = size(X_test, 1);

% X has shape [samples, time_samples, nb_electrodes]
X_train = reshape(X_train, [nb_trials_train, nb_time_samples, nb_electrodes]);
X_test = reshape(X_test, [nb_trials_test, nb_time_samples, nb_electrodes]);

% Plot channels from first sample
figure(1)
subplot(1,2,1)
plot(squeeze(X_train(plot_trial,:,:))), axis tight
title("Train sample - Unormalized");
subplot(1,2,2)
plot(squeeze(X_test(plot_trial,:,:))), axis tight
title("Test sample - Unormalized");

% Normalize each channel
for c = 1:nb_electrodes
    channel = X_train(:,:,c);
    channel_mean = mean(channel(:));
    channel_std = std(channel(:))
    
    X_train(:,:,c) = (X_train(:,:,c) - channel_mean)/channel_std;
    X_test(:,:,c) = (X_test(:,:,c) - channel_mean)/channel_std;
end

% Plot channels from first sample again
figure(2)
subplot(1,2,1)
plot(squeeze(X_train(plot_trial,:,:))), axis tight
title("Train sample - Normalized");
subplot(1,2,2)
plot(squeeze(X_test(plot_trial,:,:))), axis tight
title("Test sample - Normalized");
