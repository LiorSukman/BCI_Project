function [] = MI2_preprocess(recordingFolder,ToolboxFolder,destFolder)
%% Offline Preprocessing
% Assumes recorded using Lab Recorder.
% Make sure you have EEGLAB installed with ERPLAB & loadXDF plugins.

% [recordingFolder] - where the EEG (data & meta-data) are stored.

% Preprocessing using EEGLAB function.
% 1. load XDF file (Lab Recorder LSL output)
% 2. look up channel names - YOU NEED TO UPDATE THIS
% 3. filter data above 0.5 & below 40 Hz
% 4. notch filter @ 50 Hz
% 5. advanced artifact removal (ICA/ASR/Cleanline...) - EEGLAB functionality

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Some parameters (this needs to change according to your system):
addpath([ToolboxFolder, '\eeglab2021.1'])           % update to your own computer path
eeglab;                                     % open EEGLAB 
highLim = 40;                               % filter data under 40 Hz
lowLim = 0.5;                               % filter data above 0.5 Hz
recordingFile = strcat(recordingFolder,'\EEG.XDF');

% (1) Load subject data (assume XDF)
EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG.setname = 'MI_sub';

% (2) Update channel names - each group should update this according to
% their own openBCI setup.
EEG_chans(1,:) = 'C03';
EEG_chans(2,:) = 'C04';
EEG_chans(3,:) = 'P07';
EEG_chans(4,:) = 'P08';
EEG_chans(5,:) = 'O01';
EEG_chans(6,:) = 'O02';
EEG_chans(7,:) = 'F07';
EEG_chans(8,:) = 'F08';
EEG_chans(9,:) = 'F03';
EEG_chans(10,:) = 'F04';
EEG_chans(11,:) = 'T07';
EEG_chans(12,:) = 'T08';
EEG_chans(13,:) = 'P03';
EEG_chans(14,:) = 'P03';
EEG_chans(15,:) = 'P03';
EEG_chans(16,:) = 'P03';

%% (3) Low-pass filter
EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1);    % remove data above
EEG = eeg_checkset( EEG );
% (3) High-pass filter
EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);     % remove data under
EEG = eeg_checkset( EEG );
% (4) Notch filter - this uses the ERPLAB filter
EEG  = pop_basicfilter( EEG,  1:15 , 'Boundary', 'boundary', 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order',  180 );
EEG = eeg_checkset( EEG );

% Laplace
% C3 is surrounding by FC1, FC5, CP1, CP5:
C3 = EEG.data(1,:);
FC1 = EEG.data(4,:);
FC5 = EEG.data(6,:);
CP1 = EEG.data(8,:);
CP5 = EEG.data(10,:);

EEG.data(1,:) = C3 - 0.25*(FC1+FC5+CP1+CP5);

% C4 is surrounding by FC2, FC6, CP2, CP6:
C4 = EEG.data(2,:);
FC2 = EEG.data(5,:);
FC6 = EEG.data(7,:);
CP2 = EEG.data(9,:);
CP6 = EEG.data(11,:);

EEG.data(2,:) = C4 - 0.25*(FC2+FC6+CP2+CP6);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% (5) Add advanced artifact removal functions %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save the data into .mat variables on the computer
EEG_data = EEG.data;            % Pre-processed EEG data
EEG_event = EEG.event;          % Saved markers for sorting the data
save(strcat(destFolder,'\','cleaned_sub.mat'),'EEG_data');
save(strcat(destFolder,'\','EEG_events.mat'),'EEG_event');
save(strcat(destFolder,'\','EEG_chans.mat'),'EEG_chans');
                
end
