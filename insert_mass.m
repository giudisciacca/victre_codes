function [structuredMass, voidMass] = insert_mass(structure , mass0,inC)

    % init
    structuredMass = 0*structure;
    voidMass = 0*structure;
    % crop mass
    mass = crop_mass(mass0);
    alpha = 1;
    while any(size(mass)>(size(structure)-6))
       alpha = 0.8*alpha;
       mass = double(imresizen(double(mass0), [alpha,alpha,alpha], 'nearest' )>=0.9);
       mass = crop_mass(mass);
       
    end
    % get random position for the inclusion
    mass_disp =  1*(2* rand(1,3) -1);
    mass_disp(2) = 0.5*mass_disp(2);
    pmax = floor([0.5*(size(structuredMass,1) - size(mass,1)),...
        0.25*(size(structuredMass,1) - size(mass,1)), ...
        0.5*(size(structuredMass,3) - size(mass,3))  ]);
    c = floor(size(structure)/2 + (pmax).*mass_disp); 
    if nargin == 3
        c = inC;
    end
    voidMass = insertInVoid(voidMass, mass, c);
    structure(voidMass == 1) = 333;
    structuredMass = structure;
    return;
end


    function [cropped] = crop_mass(v)
        z =sum(sum(v,1),2);
        x =sum(sum(v,2),3);
        y =sum(sum(v,1),3);
        v(x==0,:,:) = [];
        v(:,y==0,:) = [];
        v(:,:,z==0) = [];
        cropped = v;
        return   
    end

    function out = insertInVoid(void, v,p)
        
        h_vxmax = floor(0.5*size(v,1)); 
        h_vymax = floor(0.5*size(v,2));
        h_vzmax = floor(0.5*size(v,3));
        
        x0 = p(1);
        y0 = p(2);
        z0 = p(3);
        
        X = uint16((x0-h_vxmax ):(x0-h_vxmax+size(v,1)-1));
        Y = uint16((y0-h_vymax ):(y0-h_vymax+size(v,2)-1));
        Z = uint16((z0-h_vzmax ):(z0-h_vzmax+size(v,3)-1));
        void(X+1,Y+1,Z+1) = v;
        out = void;
        return
    end
        