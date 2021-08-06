%% routine multivictre
%% generate  mass
clear
close all
tic
result_dir ='multifolder/';
%mkdir(result_dir(1:end-1));
Nin = 1;
N_lesions = 800;

seednames = extract_seednames('pc_','_crop.mhd');
for i =1:numel(seednames)%N_lesions
    [filename_cfg, seed] = randomizeCFG_mass('PROVA',seednames{i});
    massnames{i} = filename_cfg;
    seednames{i} = seed;
    cmd = ['echo launching breastMass ', num2str(i), ';set MY_CD=$SCRATCH/VICTRE/breastMass;',...
        'set OMP_NUM_THREADS=6;timeout 180 $MY_CD/breastMass -c $SCRATCH/VICTRE/',result_dir,filename_cfg,'; echo mass-done;rm *.vti;' ];
   % system(cmd)
    fid = fopen('cmdMass', 'a+'); 
    fprintf(fid, [cmd,'\n']); 
    fclose(fid);
end


%% extract mass
% for i =1:N_lesions
%     loadname= ['mass_',num2str(seednames{i}),'*raw'];    
%     loadname = dir(loadname);
%     loadname = {loadname.name};
%     mass = loadname{1};
%     ccn = str2double(strrep(strrep(mass, ['mass_',num2str(seednames{i}),'_'],''),'.raw',''));
%     
%     ccnt=ccn^3;
%     fileID = fopen(mass,'r'); 
%     Mass=reshape(fread(fileID,ccnt,'uint8=>uint8'), [ccn,ccn,ccn]); 
%     fclose(fileID)
%     figure; isosurface(Mass),axis image
% end



%% generate phantom
for i =1:N_lesions
    [filename_cfg, seed] = randomizeCFG_breast(num2str(seednames{i}),seednames{i});
    breastnames{i} = filename_cfg;
 
    cmd = ['echo launching breastPhantom ',num2str(i),';set MY_CD=$SCRATCH/VICTRE/breastPhantom;',...
    'set OMP_NUM_THREADS=16;timeout 3600 $MY_CD/breastPhantom -c $SCRATCH/VICTRE/',result_dir,'',filename_cfg,'; echo breast-done;' ];
    %system(cmd)
    fid = fopen('cmdBreast', 'a+'); 
    fprintf(fid, [cmd,'\n']); 
    fclose(fid);

end


%% compress phantoms
for i = 1:N_lesions 
    seedFile = num2str(seednames{i});
    cmd = ['echo launching breastCompress ',num2str(i),';set MY_CD=$SCRATCH/VICTRE/breastCompress;',...
        'set FEBIO_PATH=$SCRATCH/VICTRE/febio-2.8.0;',...
        'set OMP_NUM_THREADS=8;',...
        'timeout 7200 $MY_CD/breastCompress',' -s ',seedFile, '  -t 30 ', ' -f ', '$FEBIO_PATH ',' -d . -a  0;rm fe*;'] ;
%     %system(cmd)
%     fid = fopen('cmdCompress', 'a+'); 
%     fprintf(fid, [cmd,'\n']); 
%     fclose(fid);
end

%% crop phantoms

seednames = extract_seednames('pc_','.mhd');
result_dir='multifolder'
for i = 1:numel(seednames)

    %% crop 
    seedFile = num2str(seednames{i});
    cmd = ['echo launching breastCompress ',num2str(i),';set MY_CD=$SCRATCH/VICTRE/breastCrop;cd $VICTRE/',result_dir,';'...
        'timeout 60 $MY_CD/breastCrop',' -s ',seedFile, ' -g 1.0 -x 0 -y 0 -z 0 -d . '] ;
    %system(cmd)
    fid = fopen('cmdCrop', 'a+'); 
    fprintf(fid, [cmd,'\n']); 
    fclose(fid);

end
%%
toc 
