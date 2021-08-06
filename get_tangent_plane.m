%function [error] = get_tangent_plane(posx,posy,posz, Normalx,Normaly,Normalz)
function [error] = get_tangent_plane(p,pos, Normal)
    
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

    Normal = Normal/norm(Normal);
    vcross = [0, 0, 1];
    if sum(Normal(:)==vcross(:))==3
     vcross = [1,0,0];    
    end
    vorth1 = cross(Normal, vcross);
    vorth1 = vorth1/norm(vorth1);
    vorth2 = cross(vorth1, Normal);
    
    %Normal = Normal/norm(Normal);
    vorth2 = vorth2/ norm(vorth2);
    vorth1 = vorth1/ norm(vorth1);
    

    pos_plane = pos + (aa(:)*vorth1)+ (bb(:)*vorth2);
    %figure(2); hold on; scatter3(pos_plane(:,1),pos_plane(:,2),pos_plane(:,3),'*k');
    %[distances,~,~,~,~,~]=point2trimesh('Faces',p.Faces,'Vertices',p.Vertices,'QueryPoints', pos_plane);
    [distances,~,~,~,~,~]=point2trimesh('Faces',p.Faces,'Vertices',p.Vertices,'QueryPoints', pos_plane, 'algorithm','vectorize');

    error = sum(distances.^2);
    
    return
    

end