function [MUA, MUA0, structMUA, MUSP, MUSP0, structMUSP] = assign_opt(structure,structureref,valConA)
    if nargin < 3
        valConA = 2;
    end
    if numel(valConA) == 10
        valConS = valConA(end-1:end);
        valConA = valConA(1:8);
    end
    MUA = 0*repmat(structure,[1,1,1,8]);
    MUA0 = MUA;
    MUSP = MUA;
    MUSP0 = MUA;
% mua635 nm 670 nm 685 nm 785 nm 905 nm 930 nm 975 nm 1060
    lambda = [635, 670, 685, 785, 905, 930, 975, 1060];

            fat	= 1;
            skin = 2;
            glandular = 29;
            nipple = 33;
            muscle = 40; %collagen 10%
            ligament = 88;
            TDLU1 = 95;
            duct = 125;
            artery = 150;
            vein  = 225;
            
            lesion = 333;

            
            crom_fat = max([0,0,0,0,0],[1.5,1.5,1.2,0.2,0.025]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));%[oxy mu M, deoxy 0.01 muM, fat, water, collagen]
            crom_skin = max([0,0,0,0,0],[3,2,0.4,0.2,0.4]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_glandular = max([0,0,0,0,0],[2.5,2.5,0.5,0.45,0.2]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_nipple = max([0,0,0,0,0],[6,4,0.1,0.3,0.6]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_muscle = max([0,0,0,0,0],[4,6,0.4,0.2,0.4]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5)); %collagen 10%
            crom_ligament = max([0,0,0,0,0],[0,0,0.05,0.05,0.9]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_TDLU1 = max([0,0,0,0,0],[0,0,0.3,0.2,0.5]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_duct = max([0,0,0,0,0],[0,0,0.3,0.3,0.4]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            crom_artery = max([0,0,0,0,0],[7.5*1,0.5*1,0.0,0.0,0.0]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5)); % %7.51 to 9.37 mmol/L
            crom_vein  = max([0,0,0,0,0],[0.5*1,8*1,0.0,0.0,0.0]+[0.15,0.15,0.05,0.05,0.05].*randn(1,5));
            
           
            R = rand(3,1);
            R = R/sum(R(:));
            R = [1,1,1];
            crom_lesion = [ (5+10*rand) , (5+10*rand) , 0 , 0.2 , 0.8];
            
            
            
            list = {'fat','skin','glandular','nipple','muscle','ligament','TDLU1','duct','artery','vein', 'lesion'};
                        
            spectra_file = '/scratch0/NOT_BACKED_UP/gdisciac/VICTRE/spectra_polimi_percVol';
            ext_c = LoadSpectra(spectra_file,lambda,[1,1,1,1,1],[]);
            crom = crom_fat;
            for i =1:numel(list)
                structMUA.(['crom_',list{i}]) = eval(['crom_',list{i}]);
                if i >=2
                    cmd = ['crom = cat(1,crom,crom_',list{i} ,');'];
                    eval(cmd);
                end
            end
            
            for i = 1:(numel(list) - 1)
                mua = ext_c * crom(i,:)';
                idx = eval(['structureref ==',list{i}]);
%                 list{i}
%                 figure(30); plot(mua'), hold on;
%                 drawnow
%                 pause(2)
                
                for il = 1:numel(lambda)
                    IDX = 0*MUA0;
                    IDX(:,:,:, il) = idx;
                    MUA0(logical(IDX)) = mua(il);
                end
            end
            MUA = MUA0;
            i = numel(list);
            mua = ext_c * crom(i,:)';
            idx = eval(['structure ==',list{i}]);
            for il = 1:numel(lambda)
                IDX = 0*MUA;
                tmp = MUA0(:,:,:,il);
                IDX(:,:,:, il) = idx;
                if numel(valConA)==8
                    tmpmua = valConA(il);
                else
                    tmpmua = valConA*mean(tmp(logical(idx(:))));                                                    
                end
                MUA(logical(IDX)) = tmpmua;

            end
            
 
% musp            
            
            lambda0 = 620;
            Allval = unique(structure(:));
            A = 0*structure;
            B = 0*structure;
            MUSP0 = zeros(size(structure,1),size(structure,2),size(structure,3), numel(lambda));
            for val = Allval(1:end-1)'
                a = 0.5+1*rand(1);
                b = 0.2+ 0.9* rand(1);
                musp = a*(lambda/lambda0).^(-b) + (0.05*a*(lambda/lambda0).^(-b)).*randn(1,8);
                idx = (structureref==val);
                A(idx) = a; 
                B(idx) = b;
                for il = 1:numel(lambda)
                    IDX = 0*MUSP0;
                    IDX(:,:,:, il) = idx;
                    MUSP0(logical(IDX))= musp(il); 
                end
            end
                                   
            structMUSP.Abck = A;
            structMUSP.Bbck = B;
            
            
            MUSP = MUSP0;
            val = Allval(end);
            a = 0.5+1*rand(1);
            b = 0.2+ 0.9* rand(1);
            musp = a*(lambda/lambda0).^(-b) + (0.1*a)*randn(1,8);
            idx = (structure==val);
            A(idx) = a; 
            B(idx) = b;
            
            if numel(valConA)==2
                tmpmus = valConS(1)*mean(A(logical(idx(:))))*(lambda/lambda0).^(-valConS(2)) + (0.1*a)*randn(1,8);
            else
                tmpmus = musp;                                                    
            end
            
            for il = 1:numel(lambda)
                IDX = 0*MUSP0;
                IDX(:,:,:, il) = idx;
                MUSP(logical(IDX)) = tmpmus(il); 
            end
            structMUSP.Atot = A;
            structMUSP.Btot = B;
            
    return

end