%%generate, compress and breast phantom

%% breast phantom
% write cfg file

% run executable
tic 
cmd = ['echo launching breastPhantom;set MY_CD=$SCRATCH/VICTRE/breastPhantom;',...
    'set OMP_NUM_THREADS=20;$MY_CD/breastPhantom -c $SCRATCH/VICTRE/myresults/p_12345.cfg; echo breast-done' ];

system(cmd)
%system('cp p_12345.raw.gz NoDisp_p_12345.raw.gz;cp p_12345.mhd NoDisp_p_12345.mhd;gunzip p_12345.raw.gz')
toc

%mhd_phantom = read_mhd('./myresults/p_12345.mhd');

%% compress
seedFile = '12345';
direc = '$SCRATCH/VICTRE/myresults';
tic 
cmd = ['echo launching breastCompress;set MY_CD=$SCRATCH/VICTRE/breastCompress;',...
    'set FEBIO_PATH=$SCRATCH/VICTRE/febio-2.8.0;',...
    'set OMP_NUM_THREADS=12;',...
    '$MY_CD/breastCompress',' -s ',seedFile, '  -t 30 ', ' -f ', '$FEBIO_PATH',' -d ',direc,' -a 0 '] ;
toc
system(cmd)
system('rm -r fe*;rm -r p_*_*.*')
% system('gunzip pc_12345.raw')
% system('gzip pc_12345.raw')

%% crop 
cmd = ['echo launching breastCompress;set MY_CD=$SCRATCH/VICTRE/breastCrop;',...
    '$MY_CD/breastCrop',' -s ',seedFile, ' -g 1.0 -x 0 -y 0 -z 0 -d . '] ;
system(cmd)
system('gunzip ./pc_12345_crop.raw.gz')


%a = read_mhd('./pc_12345.mhd');
%%
a = read_mhd('./pc_12345_crop.mhd');
voxels = a.data;
xmax = 34/0.5; %mm/voxel size 
ymax = 44/0.5;
zmax = 30/0.5;
x0 = 1;
y0 = round(size(voxels,2)/2 - ymax/2);
z0 = 6; 
structure = voxels(x0:(x0+xmax-1),y0:(y0+ymax-1), z0:z0+zmax-1) ;
structure(structure == 0) = 2;
structure(structure == 50) = 2;
structure = permute(structure, [2,1,3]);
%% generate 
cd myresults
cmd = ['echo launching breastMass;set MY_CD=$SCRATCH/VICTRE/breastMass;',...
    'set OMP_NUM_THREADS=20;$MY_CD/breastMass -c $SCRATCH/VICTRE/myresults/testMass.cfg; echo mass-done' ];

system(cmd)
%% load mass and insert it in structure
mass = './mass_12345_64.raw';
ccn = 64;
ccnt=ccn^3;
fileID = fopen(mass,'r'); 
Mass=reshape(fread(fileID,ccnt,'uint8=>uint8'), [ccn,ccn,ccn]); 
fclose(fileID);
structure(structure==0) = 1;
%system('cp p_12345.raw.gz NoDisp_p_12345.raw.gz;cp p_12345.mhd NoDisp_p_12345.mhd;gunzip p_12345.raw.gz')
[structureMass,voidMass] = insert_mass(structure, Mass);
[structopt.mua,structopt.mua0,~,structopt.musp,structopt.musp0,~] = assign_opt(structureMass, structure,125e-3 );
contrastMua = structopt.mua - structopt.mua0;
contrastMusp = structopt.musp - structopt.musp0;
%assign_std(structureMass);
%structopt.musp = assign_opt(structure,structure);

structkwave = assign_std(structure);
dimVox = 0.5;
save('VICTRE125e-3.mat','-v7.3','structkwave','structure','structopt','structureMass','voidMass','dimVox','contrastMua','contrastMusp')
mask = voidMass;
delta = 0.5;
save('priorVICTRE.mat','-v7.3','delta','mask')

disp('saved')
%%
%voxels = voxels(:,:,airz )
%b = (cropped~=0);

b = structopt.mua(:,:,:,3);
b = contrastMua(:,:,:,1)./structopt.mua(:,:,:,1);
for i = 1:size(b,3)
    figure(1); imagesc(b(:,:,i)),colorbar,axis image,pause()
end



%% find plane
% vox = mhd_phantom.data;
% bin_vox = smooth3(vox ~= 0, 'box', [5,5,5]);
% for i =1:3
%     x{i} = linspace(mhd_phantom.origin(i),mhd_phantom.size(i)*mhd_phantom.spacing(i), mhd_phantom.size(i));
% end
% %[xx,yy,zz]=ndgrid(x{1},x{2},x{3});
% %global p
% p = (isosurface(x{2},x{1},x{3},bin_vox));
% 
% dt = bwdist(bin_vox);

% p.Vertices = p.vertices;
% p.Faces = p.faces; 
% normals = patchnormals(patch(p));
% %figure(2); quiver3(p.Vertices(:,1),p.Vertices(:,2),p.Vertices(:,3), normals(:,1), normals(:,2), normals(:,3))
% figure(2); (isosurface(x{2},x{1},x{3},bin_vox));%quiver3(p.Vertices(:,1),p.Vertices(:,2),p.Vertices(:,3), normals(:,1), normals(:,2), normals(:,3))
% axis image
% % calculate tangent plane
% %idx = find(p.Vertices(:,1)>= 0);% & p.Vertices(:,2)>= 0 & p.Vertices(:,3)>= 0);
% idx = find(p.Vertices(:,2)>= 25);
% e = zeros(1,numel(idx));
% k = 1;
% a = linspace(-20, 20, 3);
% b = linspace(-20, 20, 3);    
% [aa,bb] = ndgrid(a,b);
% 
% xV = p.Vertices(:,1);
% yV = p.Vertices(:,2);
% zV = p.Vertices(:,3);
% nx = normals(:,1);
% ny = normals(:,2);
% nz = normals(:,3);
%     idx = 1:4;
%     tic 
%     arrayfun(@get_tangent_plane,xV(idx),yV(idx),zV(idx),nx(idx),ny(idx),nz(idx))
%     toc
% Vertices = p.Vertices;
% e = get_tangent_planeVec(p,Vertices(idx,:), normals(idx,:));
% parfor k =1:numel(idx)
%     e(k) = get_tangent_plane(p,Vertices(idx(k),:), normals(idx(k),:),aa,bb);
% end
% %mypatch = patch('vertices', p.vertices, 'faces',p.faces)
%[no_x, no_y, no_z] = surfnorm(p.vertices(:,1),p.vertices(:,2),p.vertices(:,3));

%% TO DO, WRITE OPTIMISATION FUNCTION


%% generate mass
% write cfg

%














