function Mass = read_mass_mhd(seedFile)
    
    loadname= ['mass_',num2str(seedFile),'.raw']; 
    FLAG = 0;
    if ~exist(loadname,'file') && exist([loadname,'.gz'], 'file')
        FLAG = 1;
    end
    
    if FLAG == 1 
        system(['gunzip ',loadname,'.gz'] )
    end
    loadname = dir([loadname(1:end-4),'_*.raw']);
    loadname = {loadname.name};

    mass = loadname{1};
    
    ccn = str2double(strrep(strrep(mass, ['mass_',num2str(seedFile),'_'],''),'.raw',''));    
    ccnt=ccn^3;
    fileID = fopen(mass,'r'); 
    Mass=reshape(fread(fileID,ccnt,'uint8=>uint8'), [ccn,ccn,ccn]); 
    fclose(fileID)%#ok

    if FLAG == 1 
        system(['gzip ',loadname] )
    end

    
    if nargin > 1
        figure; isosurface(Mass),axis image
    end
end