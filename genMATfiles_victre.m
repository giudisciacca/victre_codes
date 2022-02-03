%%genMAT
seednames = (intersect(cell2mat(extract_seednames('mass_','.cfg'))',...
            cell2mat(extract_seednames('pc_','_crop.mhd'))'));
SAVE = 1;
k =1;
for k = 1
    
        Mass = read_mass_mhd(seedFile);
        [structure,percV(k)] = read_structure_mhd(seedFile);

        ifl = randi(2)-1;
        randAbs = -1;
        if randi(2) ==1
            % benign
           avga = [0.177,0.091,0.115,0.095,0.159,0.213,0.237,0.136]/10;
           stda = [0.015,0.01 ,0.012 ,0.01 , 0.015 , 0.013,0.011 ,0.013]/10;
           while any(randAbs <=0.002)
               randAbs = [ avga+stda.*randn(1,8)  , 1.5+0.025*randn(1,1), 0.3+0.005*randn(1,1)];
               if rand(1)<=0.25
                randAbs(end-1) = 0.3+0.01*randn(1,1);
               end
           end
           label = 'benign';
           benign =1;
        else

            avga = [0.238,0.091,0.124,0.093,0.201,0.262,0.276,0.181]/10;
            stda = [0.015,0.028 ,0.056 ,0.048 , 0.015 , 0.013,0.011 ,0.013]/10;
            while any(randAbs(:) <=0.002)
                randAbs = [ avga+stda.*randn(1,8)  , 1.4+0.025*randn(1,1), 0.3+0.005*randn(1,1)];
            end
            label = 'malignant';
            benign = 0;
        end

        [structopt.mua,structopt.mua0,~,structopt.musp,structopt.musp0,~] = assign_opt(structureMass, structure,randAbs, benign );

        contrastMua = structopt.mua - structopt.mua0;
        contrastMusp = structopt.musp - structopt.musp0;
        dimVox = 0.5;
        if SAVE
            save([foldout,filesep,'VICTRE_PARADIGM_',num2str(k),'.mat'],'-v7.3','label','structkwave','structure','structopt','structureMass','voidMass','dimVox','contrastMua','contrastMusp')
            mask = logical(voidMass);
            delta = 0.5;
            save([foldout,filesep,'Prior_VICTRE_PARADIGM',num2str(k),'.mat'],'-v7.3','delta','mask')
            disp([ 'saved: ', num2str(k) ])
        end
end
    

