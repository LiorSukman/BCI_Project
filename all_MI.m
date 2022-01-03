% ToolboxFolder - includes the EEGLAB folder
% destFolder - destination folder where all the files will be saved
% recordingFolder - folder with .xdf file
% recordingFolder should be with '\' at the end
% destFolder = recordingFolder

% recordingFolder ='C:\Recordings\Sub3\sub-P001\ses-S002\eeg\'
% ToolboxFolder = 'C:\Users\ronig\Documents\University\PhD\Courses\BCI4ALS'

function all_MI(ToolboxFolder,recordingFolder,destFolder)
MI2_preprocess(recordingFolder,ToolboxFolder,destFolder);
MI3_segmentation(destFolder);
MI4_newFeatureExtraction(destFolder);
