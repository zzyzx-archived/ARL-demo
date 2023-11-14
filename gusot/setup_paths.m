function setup_paths()

% Add the neccesary paths
pathstr= fileparts(mfilename('fullpath'));

% Tracker implementation
addpath(genpath([pathstr '/implementation/']));

% Utilities
addpath([pathstr '/utils/']);
addpath([pathstr '/myutils/']);
addpath([pathstr '/myutils/seg_utils/']);

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% PDollar toolbox
addpath(genpath([pathstr '/external_libs/pdollar_toolbox/channels']));
addpath(genpath([pathstr '/external_libs/pdollar_toolbox']));

% Mtimesx
addpath([pathstr '/external_libs/mtimesx/']);

% mexResize
addpath([pathstr '/external_libs/mexResize/']);
