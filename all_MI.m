% ToolboxFolder - includes the EEGLAB folder
% destFolder - destination folder where all the files will be saved

function all_MI(ToolboxFolder,recordingFolder,destFolder)
MI2_preprocess(recordingFolder,ToolboxFolder,destFolder);
MI3_segmentation(destFolder);
MI4_newFeatureExtraction(destFolder);
