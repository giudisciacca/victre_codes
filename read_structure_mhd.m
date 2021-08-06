function structure = read_structure_mhd(seedFile)

    
    loadname = ['./pc_',num2str(seedFile),'_crop'];
    FLAG = 0;
    if ~exist([loadname,'.raw'],'file') && exist([loadname,'.raw.gz'], 'file')
        FLAG = 1;
    end
    
    if FLAG == 1 
        system(['gunzip ',loadname,'.raw.gz'] )
    end

    a = read_mhd([loadname,'.mhd']);
    
    voxels = a.data;
    xmax = 54/0.5; %mm/voxel size 
    ymax = 44/0.5;
    zmax = 45/0.5;
    %voxels = cat(1,voxels,voxels(end:-1:10,:,:));
    
    x0 = round(size(voxels,2)/2 - xmax/2) -8;
    y0 = 1;
    z0 = 4; 
    if ((y0+ymax-1) - size(voxels,1)) > 0
        voxels = cat(1,voxels((((y0+ymax-1) - size(voxels,1))):-1:1,:,:),voxels);
    end
    
    structure = voxels(y0:(y0+ymax-1),x0:(x0+xmax-1), z0:z0+zmax-1) ;

%     sum(structure(:) == 0)
%     sum(structure(:) == 50)
    mode(structure(structure(:)~=0));
    [n,h] = hist(structure(structure(:)~=0),unique(structure(structure(:)~=0)));
    [~,idx] = sort(-n);
    h = h(idx);
    r = randi(2)-1;
    structure(structure==0) = r*h(1) +( 1-r)*h(2); %most present tissue
    
    structure(structure == 50) = 40; % chest is muscle
    structure = permute(structure, [2,1,3]);
    if FLAG == 1 
        system(['gzip ',loadname,'.raw'] )
    end
    return


end