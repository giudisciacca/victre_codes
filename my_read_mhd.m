function [img, info]=my_read_mhd(filename)
    % This function is based upon "read_mhd" function from the package
    % ReadData3D_version1 from the matlab exchange.
    % Copyright (c) 2010, Dirk-Jan Kroon
    % [image info ] = read_mhd(filename)
    FLAG = 0;
    if exist([filename,'.raw.gz'],'file')
        FLAG = 1;
    end

    if FLAG == 1
        system(['gunzip ',filename,'.raw.gz'])
    end

    [img,info] = read_mhd([filename,'.mhd']);

    if FLAG == 1
        system(['gunzip ',filename,'.raw'])
    end

    return

end