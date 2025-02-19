function [filename,seed]= randomizeCFG_mass(stri,seed )

if ~exist('seed','var') || isempty(seed)
        seed = randi(1e6);
end



stri = 'PROVA';
%% phantom
targetFatFrac= 0.05+0.85*rand;
% # bottom scale
a1b=1+ 0.5*rand;
% # top scale
a1t=1+ 0.5*rand;
% # left scale
a2l=1+ 0.5*rand;
% # right scale
a2r=1+ 0.5*rand;
% # outward scale
a3=1+ 2*rand;
% # u quadric exponent
eps1=1.05+0.25*rand;
% # minimum scale skin seeds in nipple direction
minSkinScaleNippleDir=5.0+5*rand;
% # maximum scale skin seeds in nipple direction
maxSkinScaleNippleDir=10.0+5*rand;
% # back strength
backStrength=1.0+5*rand;
% # fraction of branch length per segment
segFrac=0.15+0.1*rand;

vec_val = [targetFatFrac;
seed;
a1b;
a1t;
a2l;
a2r;
a3;
eps1;
minSkinScaleNippleDir;
maxSkinScaleNippleDir;
backStrength;
segFrac;];

sting2sprintf = ['##########################\n',...
'# breast phantom configuration file\n',...
'##########################\n',...
'#####################\n',...
'# basic variables\n',...
'#####################\n',...
'[base]\n',...
'# output directory\n',...
'outputDir=/scratch0/NOT_BACKED_UP/gdisciac/VICTRE/multifolder/\n',...
'# phantom voxel size (mm)\n',...
'imgRes=0.5\n',...
'# thickness of breast skin (mm)\n',...
'skinThick=0.75\n',...
'# nipple length (mm)\n',...
'nippleLen=4.0\n',...
'# nipple radius (mm)\n',...
'nippleRad=4.0\n',...
'# nipple radius (mm)\n',...
'areolaRad=8.0\n',...
'# left breast - select left or right breast (boolean)\n',...
'leftBreast=true\n',...
'# desired fat fraction\n',...
'targetFatFrac=%g\n',...
'# random number seed (unsigned int)\n',...
'# chosen randomly if not set\n',...
'seed=%d\n',...
'#####################\n',...
'# breast surface shape\n',...
'#####################\n',...
'[shape]\n',...
'# u resolution of base shape\n',...
'ures=0.005\n',...
'# v resolution of base shape\n',...
'vres=0.005\n',...
'# minimum point separation (mm)\n',...
'pointSep=0.005\n',...
'# back ring thickness (mm)\n',...
'ringWidth=10.0\n',...
'# back ring step size (mm)\n',...
'ringSep=0.5\n',...
'# angle to preserve while smoothing (degrees)\n',...
'featureAngle=20.0\n',...
'# fraction of triangles to decimate\n',...
'targetReduction=0.05\n',...
'# bottom scale\n',...
'a1b=%g\n',...
'# top scale\n',...
'a1t=%g\n',...
'# left scale\n',...
'a2l=%g\n',...
'# right scale\n',...
'a2r=%g\n',...
'# outward scale\n',...
'a3=%g\n',...
'# u quadric exponent\n',...
'eps1=%g\n',...
'# v quadric exponent\n',...
'eps2=1.0\n',...
'# do ptosis deformation (boolean)\n',...
'doPtosis=true\n',...
'ptosisB0=0.2\n',...
'ptosisB1=0.05\n',...
'# do turn deformation (boolean)\n',...
'doTurn=false\n',...
'turnC0=-0.498\n',...
'turnC1=0.213\n',...
'# do top shape deformation (boolean)\n',...
'doTopShape=true\n',...
'topShapeS0=0.0\n',...
'topShapeS1=0.0\n',...
'topShapeT0=-12.0\n',...
'topShapeT1=-5.0\n',...
'# do flatten size deformation (boolean)\n',...
'doFlattenSide=true\n',...
'flattenSideG0=1.5\n',...
'flattenSideG1=-0.5\n',...
'# do turn top deformation (boolean)\n',...
'doTurnTop=true\n',...
'turnTopH0=0.166\n',...
'turnTopH1=-0.372\n',...
'#####################\n',...
'# breast compartment\n',...
'#####################\n',...
'[compartments]\n',...
'# number of breast compartments\n',...
'num=10\n',...
'# distance along nipple line of compartment seed base (mm)\n',...
'seedBaseDist=16\n',...
'# fraction of phantom in nipple direction forced to be fat\n',...
'backFatBufferFrac=0.008\n',...
'# number of backplane seed points\n',...
'numBackSeeds=150\n',...
'# maximum seed jitter (fraction of subtended angle)\n',...
'angularJitter=0.125\n',...
'# maximum seed jitter in nipple direction (mm)\n',...
'zJitter=5.0\n',...
'# maximum radial distance from base seed as a fraction of distance to breast surface\n',...
'maxFracRadialDist=0.5\n',...
'# minimum radial distance from base seed as a fraction of distance to breast surface\n',...
'minFracRadialDist=0.25\n',...
'# minimum scale in nipple direction\n',...
'minScaleNippleDir=0.01\n',...
'# maximum scale in nipple direction\n',...
'maxScaleNippleDir=0.01\n',...
'# minimum scale in non-nipple direction\n',...
'minScale=30.0\n',...
'# maximum scale in non-nipple direction\n',...
'maxScale=40.0\n',...
'# minimum gland strength\n',...
'minGlandStrength=30.0\n',...
'# maximum gland strength\n',...
'maxGlandStrength=30.0\n',...
'# maximum compartment deflection angle from pointing towards nipple (fraction of pi)\n',...
'maxDeflect=0.01\n',...
'# minimum scale skin seeds in nipple direction\n',...
'minSkinScaleNippleDir=%g\n',...
'# maximum scale skin seeds in nipple direction\n',...
'maxSkinScaleNippleDir=%g\n',...
'# minimum scale skin in non-nipple direction\n',...
'minSkinScale=200.0\n',...
'# maximum scale skin in non-nipple direction\n',...
'maxSkinScale=400.0\n',...
'# skin strength\n',...
'skinStrength=0.5\n',...
'# back scale\n',...
'backScale=60.0\n',...
'# back strength\n',...
'backStrength=%g\n',...
'# nipple scale\n',...
'nippleScale=5.0\n',...
'# nipple strength\n',...
'nippleStrength=10.0\n',...
'# check seeds within radius (mm)\n',...
'voronSeedRadius=100.0\n',...
'#####################\n',...
'# TDLU variables\n',...
'#####################\n',...
'[TDLU]\n',...
'# maximum TDLU length\n',...
'maxLength=2.0\n',...
'# minimum TDLU length\n',...
'minLength=1.0\n',...
'# maximum TDLU width\n',...
'maxWidth=1.0\n',...
'# minimum TDLU width\n',...
'minWidth=0.5\n',...
'#####################\n',...
'# Perlin noise variables\n',...
'#####################\n',...
'[perlin]\n',...
'# maximum fraction of radius deviation \n',...
'maxDeviation=0.1\n',...
'# starting frequency\n',...
'frequency=0.1\n',...
'# octave frequency multiplier\n',...
'lacunarity=2.0\n',...
'# octave signal decay\n',...
'persistence=0.5\n',...
'# number of frequency octaves\n',...
'numOctaves=6\n',...
'# x direction noise generation seed\n',...
'xNoiseGen=683\n',...
'# y direction noise generation seed\n',...
'yNoiseGen=4933\n',...
'# z direction noise generation seed\n',...
'zNoiseGen=23\n',...
'# seed noise generation\n',...
'seedNoiseGen=3095\n',...
'# shift noise generation seed\n',...
'shiftNoiseGen=11\n',...
'#####################\n',...
'# Compartment boundary noise\n',...
'#####################\n',...
'[boundary]\n',...
'# maximum fraction of distance deviation \n',...
'maxDeviation=0.1\n',...
'# starting frequency\n',...
'frequency=0.15\n',...
'# octave frequency multiplier\n',...
'lacunarity=1.5\n',...
'# octave signal decay\n',...
'persistence=0.5\n',...
'#####################\n',...
'# Lobule boundary perturbation noise\n',...
'#####################\n',...
'[perturb]\n',...
'# maximum fraction of distance deviation \n',...
'maxDeviation=0.25\n',...
'# starting frequency\n',...
'frequency=0.09\n',...
'# octave frequency multiplier\n',...
'lacunarity=2.0\n',...
'# octave signal decay\n',...
'persistence=0.4\n',...
'#####################\n',...
'# Lobule glandular buffer noise\n',...
'#####################\n',...
'[buffer]\n',...
'# maximum fraction of distance deviation \n',...
'maxDeviation=0.15\n',...
'# starting frequency\n',...
'frequency=0.05\n',...
'# octave frequency multiplier\n',...
'lacunarity=1.5\n',...
'# octave signal decay\n',...
'persistence=0.5\n',...
'#####################\n',...
'# Voronoi segmentation variables\n',...
'#####################\n',...
'[voronoi]\n',...
'# fat voronoi seed density (mm^-3)\n',...
'fatInFatSeedDensity=0.001\n',...
'# fat voronoi seed in glandular tissue density (mm^-3)\n',...
'fatInGlandSeedDensity=0.001\n',...
'# glandular voronoi seed density (mm^-3)\n',...
'glandInGlandSeedDensity=0.0005\n',...
'# maximum deflection (fraction of pi)\n',...
'TDLUDeflectMax=0.15\n',...
'# minimum length scale\n',...
'minScaleLenTDLU=0.1\n',...
'# maximum length scale\n',...
'maxScaleLenTDLU=0.2\n',...
'# minimum width scale\n',...
'minScaleWidTDLU=40.0\n',...
'# maximum width scale\n',...
'maxScaleWidTDLU=45.0\n',...
'# minimum strength\n',...
'minStrTDLU=20.0\n',...
'# maximum strength\n',...
'maxStrTDLU=22.0\n',...
'# maximum deflection (fraction of pi)\n',...
'fatInFatDeflectMax=0.15\n',...
'# minimum length scale\n',...
'minScaleLenFatInFat=5.0\n',...
'# maximum length scale\n',...
'maxScaleLenFatInFat=10.0\n',...
'# minimum width scale\n',...
'minScaleWidFatInFat=50.0\n',...
'# maximum width scale\n',...
'maxScaleWidFatInFat=60.0\n',...
'# minimum strength\n',...
'minStrFatInFat=40.0\n',...
'# maximum strength\n',...
'maxStrFatInFat=50.0\n',...
'# maximum deflection (fraction of pi)\n',...
'fatInGlandDeflectMax=0.15\n',...
'# minimum length scale\n',...
'minScaleLenFatInGland=1.0\n',...
'# maximum length scale\n',...
'maxScaleLenFatInGland=2.0\n',...
'# minimum width scale\n',...
'minScaleWidFatInGland=30.0\n',...
'# maximum width scale\n',...
'maxScaleWidFatInGland=40.0\n',...
'# minimum strength\n',...
'minStrFatInGland=20.0\n',...
'# maximum strength\n',...
'maxStrFatInGland=22.0\n',...
'# maximum deflection (fraction of pi)\n',...
'glandInGlandDeflectMax=0.15\n',...
'# minimum length scale\n',...
'minScaleLenGlandInGland=1.0\n',...
'# maximum length scale\n',...
'maxScaleLenGlandInGland=2.0\n',...
'# minimum width scale\n',...
'minScaleWidGlandInGland=30.0\n',...
'# maximum width scale\n',...
'maxScaleWidGlandInGland=40.0\n',...
'# minimum strength\n',...
'minStrGlandInGland=20.0\n',...
'# maximum strength\n',...
'maxStrGlandInGland=22.0\n',...
'# check seeds in radius (mm) \n',...
'seedRadius=40.0\n',...
'#####################\n',...
'# fat variables\n',...
'#####################\n',...
'[fat]\n',...
'# min lobule axis length (mm)\n',...
'minLobuleAxis=20.0\n',...
'# max lobule axis length (mm)\n',...
'maxLobuleAxis=30.0\n',...
'# axial ratio min\n',...
'minAxialRatio=0.13\n',...
'# axial ratio max\n',...
'maxAxialRatio=0.75\n',...
'# minimum ligament separation between lobules\n',...
'minLobuleGap=0.15\n',...
'# maximum of absolute value of Fourier coefficient as fraction of main radius\n',...
'maxCoeffStr=0.1\n',...
'# minimum of absolute value of Fourier coefficient as fraction of main radius\n',...
'minCoeffStr=0.05\n',...
'# maximum number of trial lobules\n',...
'maxLobuleTry=401\n',...
'#####################\n',...
'# ligament variables\n',...
'#####################\n',...
'[lig]\n',...
'thickness=0.1\n',...
'targetFrac=0.85\n',...
'maxTry=15000\n',...
'minAxis=20.0\n',...
'maxAxis=25.0\n',...
'minAxialRatio=0.2\n',...
'maxAxialRatio=0.3\n',...
'maxPerturb=0.05\n',...
'maxDeflect=0.12\n',...
'scale=0.007\n',...
'lacunarity=1.5\n',...
'persistence=0.3\n',...
'numOctaves=6\n',...
'#####################\n',...
'# duct tree variables\n',...
'#####################\n',...
'[ductTree]\n',...
'# target number of branches (uint)\n',...
'maxBranch=400\n',...
'# maximum generation (uint)\n',...
'maxGen=7\n',...
'# initial radius of tree (mm)\n',...
'initRad=0.5\n',...
'# base Length of root duct at nipple (mm)\n',...
'baseLength=7.6\n',...
'# number of voxels for tree density tracking (uint)\n',...
'nFillX=50\n',...
'nFillY=50\n',...
'nFillZ=50\n',...
'#####################\n',...
'# duct branch variables\n',...
'#####################\n',...
'[ductBr]\n',...
'# minimum branch radius to have children (mm)\n',...
'childMinRad=0.1\n',...
'# minimum starting radius as a fraction of parent end radius\n',...
'minRadFrac=0.65\n',...
'# maximum starting radius as a fraction of parent end radius\n',...
'maxRadFrac=0.99\n',...
'# length reduction as fraction of parent length\n',...
'lenShrink=0.5\n',...
'# maximum jitter in branch length (fraction)\n',...
'lenRange=0.1\n',...
'# aximuthal angle noise (radians)\n',...
'rotateJitter=0.1\n',...
'#####################\n',...
'# duct segment variables\n',...
'#####################\n',...
'[ductSeg]\n',...
'# radius distribution shape parameters\n',...
'radiusBetaA=6.0\n',...
'radiusBetaB=10.0\n',...
'# fraction of branch length per segment\n',...
'segFrac=0.25\n',...
'# maximum radius of curvature (mm)\n',...
'maxCurvRad=10.0\n',...
'# maximum length of segment based on\n',...
'# curvature (fraction of pi radians)\n',...
'maxCurvFrac=0.5\n',...
'# min and max end radius as fraction of start radius\n',...
'minEndRad=0.95\n',...
'maxEndRad=1.0\n',...
'# cost function preferential angle weighting\n',...
'angleWt=1.0\n',...
'# cost function density weighting\n',...
'densityWt=20.0\n',...
'# number of trial segments to generate (uint)\n',...
'numTry=50\n',...
'# maximum number of segments to generate before\n',...
'# giving up and reducing length (uint)\n',...
'maxTry=100\n',...
'# total number of segment tries before completely giving up\n',...
'absMaxTry=10000\n',...
'# step size for checking segment is valid (mm)\n',...
'roiStep=0.1\n',...
'####################\n',...
'# vessel tree variables\n',...
'#####################\n',...
'[vesselTree]\n',...
'# target number of branches (uint)\n',...
'maxBranch=750\n',...
'# maximum generation (uint)\n',...
'maxGen=6\n',...
'# initial radius of tree (mm)\n',...
'initRad=0.75\n',...
'# base length of root vessel (mm)\n',...
'baseLength=15.0\n',...
'# number of voxels for tree density tracking (uint)\n',...
'nFillX=30\n',...
'nFillY=69\n',...
'nFillZ=69\n',...
'#####################\n',...
'# vessel branch variables\n',...
'#####################\n',...
'[vesselBr]\n',...
'# minimum branch radius to have children (mm)\n',...
'childMinRad=0.1\n',...
'# minimum starting radius as a fraction of parent end radius \n',...
'minRadFrac=0.65\n',...
'# maximum starting radius as a fraction of parent end radius \n',...
'maxRadFrac=0.99\n',...
'# length reduction as fraction of parent length\n',...
'lenShrink=0.8\n',...
'# maximum jitter in branch length (fraction)\n',...
'lenRange=0.1\n',...
'# aximuthal angle noise (radians)\n',...
'rotateJitter=0.1\n',...
'#####################\n',...
'# vessel segment variables\n',...
'#####################\n',...
'[vesselSeg]\n',...
'# radius distribution shape parameters\n',...
'radiusBetaA=6.0\n',...
'radiusBetaB=10.0\n',...
'# fraction of branch length to segment\n',...
'segFrac=%g\n',...
'# maximum radius of curvature (mm)\n',...
'maxCurvRad=200.0\n',...
'# maximum length of segment based on \n',...
'# curvature (fraction of pi radians)\n',...
'maxCurvFrac=0.5\n',...
'# min and max end radius as fraction of start radius\n',...
'minEndRad=0.95\n',...
'maxEndRad=1.0\n',...
'# cost function preferential angle weighting\n',...
'angleWt=100.0\n',...
'# cost function density weighting\n',...
'densityWt=1.0\n',...
'# cost function direction weighting\n',...
'dirWt = 100.0\n',...
'# number of trial segments to generate (uint)\n',...
'numTry=100\n',...
'# maximum number of segments to generate before\n',...
'# giving up and reducing length (uint)\n',...
'maxTry=300 \n',...
'# total number of segment tries before completely giving up\n',...
'absMaxTry=100000\n',...
'# step size for checking segment is valid (mm)\n',...
'roiStep=0.1'];

string = sprintf(sting2sprintf,vec_val);
filename = ['breast_',num2str(seed), '.cfg'];
fileID = fopen(filename, 'w');
fprintf(fileID,string);
fclose(fileID);

end