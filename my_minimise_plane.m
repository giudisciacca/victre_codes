function [x, arrayx] = my_minimise_plane( Vertices, Faces, Normals)

    global Vertices
    % def functions
    func = @(x) get_tangent_planeVec_vorth(x,Vertices,Faces,Normals);
    % find initial vertex
    
    x0idx = find(Vertices(:,1)>30 & Vertices(:,2)>20&Vertices(:,3)>30);
    x0 = [Vertices(x0idx(1),1),Vertices(x0idx(1),2),Vertices(x0idx(1),3)];
    
    x = x0;
    arrayx = zeros(50,3);
    for i = 1:50
        i
        [L0, vorth1, vorth2] = func(x);
%         if i > 1 && (L0old - L0) < 0.000001* L0old
%             break
%         end
        L0old = L0;
        vorth1 = 1* vorth1;
        vorth2 = 1* vorth2;
        alpha1 = 1;
        alpha2 = 2;
        [~,dL1] = get_gradient(x, vorth1, alpha1, L0);
        [~,dL2] = get_gradient(x, vorth2, alpha2, L0);
        dL1;
        % backtrack
        [alpha1] = backtrack(x,vorth1,dL1,L0);
        [alpha2] = backtrack(x,vorth2,dL2,L0);
        x = x - alpha1 * vorth1 *dL1 - alpha2*vorth2*dL2;
        x = project2surface(x, Vertices);
        arrayx(i,:) = x
    end
   disp(x)
    function [c,ceq] = constrain(x,vert)
     c = [];
     ceq = double(ismember(x,vert))-1;            
    end
    function [L,dL] = get_gradient(x,vorth,alpha,L0)
        L = func(project2surface(x + alpha*vorth, Vertices));
        dL = (L - L0);
        dL = dL/abs(dL);
    end
    
    function out = project2surface(x, vert)
        [~,I] = min(sum((x -vert).^(2),2));
        out = vert(I,:);
    end
    function alpha = backtrack(x,vorth,dL, L0)
     alpha = 10/0.9;
     k = 1;
     L = L0 +1;
     %[L,dL] = get_gradient(x,vorth, alpha, L0);
     while L >= L0 && k < 20
         alpha = alpha*0.9;
         x_update = x - dL * vorth * alpha;
         L = (func(x_update));
         k = k + 1;
         %L = L0 - dL;
     end
    if k == 50
     alpha = 0;
    end
    end
        
end