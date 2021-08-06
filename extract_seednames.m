function sout = extract_seednames(init, fin)

s = dir([init,'*',fin]);
s = {s.name};
    for i = 1:numel(s)
        sout{i} = str2double(strrep(strrep(strrep(s{i},init,''),fin,''),'_crop',''));
    end
end