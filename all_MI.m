% ToolboxFolder - includes the EEGLAB folder
% destFolder - destination folder where all the files will be saved

function SelectedFeatureNames = all_MI(ToolboxFolder, recordingFolder, destFolder)
MI2_preprocess(recordingFolder, ToolboxFolder, destFolder);
MI3_segmentation(destFolder);
SelectedFeatureNames = MI4_featureExtraction(destFolder);
