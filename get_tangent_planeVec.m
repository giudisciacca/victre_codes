%function [error] = get_tangent_planeVec(posx,posy,posz, Normalx,Normaly,Normalz)
function [error] = get_tangent_planeVec(p,pos, Normal)
    
    %global p
%     idxvertex = [Face(1), Face(2), Face(3)];
%     
%     posv1 = p.Vertices(idxvertex(1),:);
%     posv2 = p.Vertices(idxvertex(2),:);
%     posv3 = p.Vertices(idxvertex(3),:);
%      pos = (posv1+posv2+posv3)/3;
   %
%     pos = [posx,posy,posz];
%     Normal=[Normalx,Normaly, Normalz];
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
    
    %Normal = Normal/norm(Normal);
    vorth2 = vorth2./ mynorm(vorth2);
    %vorth1 = vorth1/ mynorm(vorth1);
    samp = 5;
    %sampSQ = %samp^2;
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
%     %[distances,~,~,~,~,~]=point2trimesh('Faces',p.Faces,'Vertices',p.Vertices,'QueryPoints', pos_plane);
%     totPoints = size(pos_plane,1);
%     error = [];
%     outDist = [];
%     chunks = 500;
%     cycles = (totPoints - mod(totPoints,chunks))/chunks
%     for cyc = 1:(cycles-1)
%         cyc
%         queryPos = (((cyc-1)*chunks) + 1):((cyc)*chunks);
%         %to_query = pos_plane(queryPos',:)
%         [distances]=point2trimesh('Faces',p.Faces,'Vertices',p.Vertices,'QueryPoints', pos_plane(queryPos',:), 'Algorithm','vectorized');
%         outDist = cat(1,outDist,reshape(distances,[chunks,1]));
%         %outDist = reshape(distances, [chunks, 1]);
%         %error = cat(1, error,(sum(outDist.^2,2)));
%     end
%         queryPos = ((cycles*chunks)+1):totPoints;
%         [distances]=point2trimesh('Faces',p.Faces,'Vertices',p.Vertices,'QueryPoints', pos_plane(queryPos',:), 'Algorithm','vectorized');
%         %outDist = reshape(distances, [mod(totPoints,chunks), sampSQ]);
%         outDist = cat(1,outDist,reshape(distances,[mod(totPoints,chunks),1]));
%         outDist = reshape(outDist, [siz, sampSQ]);
%         error = (sum(outDist.^2,2));
        
    return
    
    function out = mynorm(v)
        out = sqrt(sum(v.^2,2));
        return
    end
end