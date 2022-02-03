%% This is an example of how to generate the 2 mat files to be used in the simulation and reconstruction process
% BreastPhantom, BreastMass, BreastCompress and BreastCrop from the software suite "VICTRE" need to be installed

% generate random configuration file for breast
[~,~] = randomizeCFG_breast('PHANTOM');

% generate configuration file for mass
[~,~] = randomizeCFG_mass('MASS');

% find output files from breastPhantom, compress
routine_multiVICTRE

% assign optical and acoustic properties to all generated phantoms, insert lesions, save phantom and lesion morphology to be used in reconstruction








 


 
