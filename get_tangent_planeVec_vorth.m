%function [error] = get_tangent_planeVec(posx,posy,posz, Normalx,Normaly,Normalz)
function [error, vorth1,vorth2] = get_tangent_planeVec_min(pos,Vertices, Faces,Normals)
    
    [isthere,LocResult] = ismember(pos,Vertices,'rows');
    %pos
    if isthere
        Normal=[Normals(LocResult,1),Normals(LocResult,2), Normals(LocResult,3)];
        extrad = 0;
        %pos
        %extrad
    else
        tmp = sum((pos - Vertices).^2,2); 
        [extrad,I] = min(tmp);
        %I
        %pos = [Vertices(I,1),Vertices(I,2),Vertices(I,3)];
        Normal=[Normals(I,1),Normals(I,2), Normals(I,3)];
        %pos
        %extrad =;
    end
    siz = size(Normal,1);
    Normal = Normal./mynorm(Normal);
    vcross = repmat([0, 0, 1], [siz,1,1]);
    idxVcross = find(sum(Normal == vcross,1)==3); 
    if ~isempty(idxVcross)
        vcross(idxVcross,:) = [1,0,0];     
    end
    vorth1 = cross(Normal, vcross);
    vorth1 = vorth1./mynorm(vorth1);
    vorth2 = cross(vorth1, Normal);
    vorth2 = vorth2./ mynorm(vorth2);

    samp = 3;

    R = 20;
    a = linspace(-R, R, samp);
    b = linspace(-R, R, samp);    
    [aa,bb] = ndgrid(a,b);
    valid = sqrt(aa.^2 + bb.^2) <= R;
    aa = aa(valid(:));
    bb = bb(valid(:));
    sampSQ = size(aa,1);
    aa = reshape(repmat(aa(:)', [siz,1,1]),[siz,1,sampSQ]);
    bb = reshape(repmat(bb(:)', [siz,1,1]),[siz,1,sampSQ]);
    
    
    
    pos_plane = pos + (aa.*vorth1)+ (bb.*vorth2);
    pos_plane = permute(pos_plane, [1,3,2]);
    pos_plane = reshape(pos_plane, [siz*size(pos_plane,2), size(pos_plane,3)]);

%figure(2); hold on; scatter3(pos_plane(:,1),pos_plane(:,2),pos_plane(:,3),'*k');
   [distances]=point2trimesh('Faces',Faces,'Vertices',Vertices,'QueryPoints', pos_plane, 'algorithm','vectorized');
    outDist = reshape(distances, [siz, sampSQ]);
    error = sqrt(mean(sum(outDist.^2,2)));%+extrad
    return
    
    function out = mynorm(v)
        out = sqrt(sum(v.^2,2));
        return
    end
end