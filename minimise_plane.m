function minimise_plane( Vertices, Faces, Normals)


    % def functions
    func = @(x) get_tangent_planeVec_min(x,Vertices,Faces,Normals);
    % find initial vertex
    
    A = []; % No other constraints
    b = []; 
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    nonlcon = @(x) constrain(x, Vertices);
    
    x0idx = find(Vertices(:,1)>30 & Vertices(:,2)>20&Vertices(:,3)>30);
    x0 = [Vertices(x0idx(1),1),Vertices(x0idx(1),2),Vertices(x0idx(1),3)];
    
    %x = fmincon(func,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    options = optimoptions('fminunc','FiniteDifferenceStepSize',[0.01,0.01,0.01]);
    %options.FiniteDifferenceStepSize = [3,3,3];
    x = fminunc(func,x0, options);
   disp(x)
    function [c,ceq] = constrain(x,vert)
     c = [];
     ceq = double(ismember(x,vert))-1;            
    end


end