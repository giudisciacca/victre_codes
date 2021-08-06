seednames = (intersect(cell2mat(extract_seednames('mass_','.cfg'))',...
            cell2mat(extract_seednames('pc_','_crop.mhd'))'));
 
k =1;
for i = [1:492,494:numel(seednames)]%Nin:Nin+N_lesions 
    i
    
    seedFile = num2str(seednames(i));
    if ~isempty(dir(['mass_',seedFile,'_*.raw']))
        Mass = read_mass_mhd(seedFile);
        structure = read_structure_mhd(seedFile);

        [structureMass,voidMass] = insert_mass(structure, Mass);
         ifl = randi(2)-1;
%         randAbs = ifl * (0.01+(0.5 - 0.01)*rand)+ (1 - ifl) * (1.5+(5 - 1.5)*rand);
        label = 'none';
        randAbs = -1;
        if randi(2) ==1
            % benign
           avga = [0.177,0.091,0.115,0.095,0.159,0.213,0.237,0.136]/10;
           stda = [0.015,0.01 ,0.012 ,0.01 , 0.015 , 0.013,0.011 ,0.013]/10;
           while any(randAbs <=0.002)
               randAbs = [ avga+stda.*randn(1,8)  , 1.5+0.15*randn(1,1), 1.2+0.07*randn(1,1)];
           end
           label = 'benign';
            plot(randAbs(1:8), '-*')
        else
            % malignant
            avga = [0.238,0.091,0.124,0.093,0.201,0.262,0.276,0.181]/10;
            stda = [0.015,0.028 ,0.056 ,0.048 , 0.015 , 0.013,0.011 ,0.013]/10;
            while any(randAbs(:) <=0.002)
                randAbs = [ avga+stda.*randn(1,8)  , 1.2+0.3*randn(1,1), 1.2+0.07*randn(1,1)];
            end
            label = 'malignant';
            plot(randAbs(1:8), '-^'),hold on
        end

        [structopt.mua,structopt.mua0,~,structopt.musp,structopt.musp0,~] = assign_opt(structureMass, structure,randAbs );
        %figure(1)
        plot(squeeze(mean(mean(mean(structopt.mua0,1),2),3)))
        hold on
         %plot(randAbs(1:8), '-*')
        drawnow
        contrastMua = structopt.mua - structopt.mua0;
        contrastMusp = structopt.musp - structopt.musp0;
        %assign_std(structureMass);
        %structopt.musp = assign_opt(structure,structure);

        structkwave = assign_std(structure);
        dimVox = 0.5;
        save(['VICTRE_PARADIGM_',num2str(k),'.mat'],'-v7.3','label','structkwave','structure','structopt','structureMass','voidMass','dimVox','contrastMua','contrastMusp')
        %save(['VICTRE_',num2str(k),'.mat'],'-v7.3','structkwave','structure','structopt','structureMass','voidMass','dimVox','contrastMua','contrastMusp')

        mask = logical(voidMass);
        delta = 0.5;
        save(['Prior_VICTRE_PARADIGM',num2str(k),'.mat'],'-v7.3','delta','mask')
        %save(['Prior_VICTRE_',num2str(k),'.mat'],'-v7.3','delta','mask')
        k = k+1;
        disp('saved')
    end
    
end
