function S = assign_std(structure)

    val = unique(structure);
    s = linspace(0,1, numel(val));
    s = s(randperm(numel(s)));
    S = 0*structure;
    for i = 1:numel(val)
       S(structure==val(i)) = s(i); 
    end
    return




end