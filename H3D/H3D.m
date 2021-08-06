%M.Sc. Guillermo García Jiménez
%Circles detection using general method Hough. The method uses a acumulator
%matrix 3D for votting.

function varargout = H3D(varargin)
% H3D MATLAB code for H3D.fig
% Last Modified by GUIDE v2.5 25-Nov-2017 02:28:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @H3D_OpeningFcn, ...
    'gui_OutputFcn',  @H3D_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
clc
% End initialization code - DO NOT EDIT


% --- Executes just before H3D is made visible.
function H3D_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for H3D
handles.output = hObject;
clc
%table
set(handles.tabla_radios,'Data',cell(2,3));
%axes Matriz_Acc
cla(handles.Matriz_Acc);
title(handles.Matriz_Acc,['Accumulator Matrix']);
handles.Matriz_Acc.FontSize=14
set(handles.Matriz_Acc, 'Xlim',[1 500], 'Ylim', [1 500], 'Xgrid', 'on', 'Ygrid', 'on', 'Zlim', [1 500], 'Zgrid', 'on','units','normalized');
handles.Matriz_Acc.XLabel.String='k';
handles.Matriz_Acc.XLabel.FontSize=14
handles.Matriz_Acc.YLabel.String='h';
handles.Matriz_Acc.YLabel.FontSize=14
handles.Matriz_Acc.ZLabel.String='number of votes';
handles.Matriz_Acc.ZLabel.FontSize=14
colorbar(handles.Matriz_Acc,'off');
%axes hist
cla(handles.histograma, 'reset');
set(handles.histograma, 'Xlim',[1 100], 'Ylim', [1 100], 'Xgrid', 'on', 'Ygrid', 'on','units','normalized');
handles.histograma.XLabel.String='radius';
handles.histograma.XLabel.FontSize=14
handles.histograma.YLabel.String='number of votes';
handles.histograma.YLabel.FontSize=14
%
cla(handles.Hough,'reset');
set(handles.Hough,'Xticklabel',[],'Yticklabel',[]);
%
cla(handles.Imagen_Original,'reset');
set(handles.Imagen_Original,'Xticklabel',[],'Yticklabel',[]);
% Update handles structure
Mostrar=2;
handles.Mostrar=Mostrar;
guidata(hObject, handles);

% UIWAIT makes H3D wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = H3D_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in importar_imagen.
function importar_imagen_Callback(hObject, eventdata, handles)
clc
handles = guidata(hObject);
%counter
handles.counter=0;
%table
set(handles.tabla_radios,'Data',cell(2,3));
%axes Imagen_Original
cla(handles.Imagen_Original,'reset');
set(handles.Imagen_Original,'Xticklabel',[],'Yticklabel',[]);
%axes Hough
cla(handles.Hough,'reset');
set(handles.Hough,'Xticklabel',[],'Yticklabel',[]);
%axes Matriz_Acc
cla(handles.Matriz_Acc);
colorbar(handles.Matriz_Acc,'off');
title(handles.Matriz_Acc,['Accumulator Matrix ']);
set(handles.Matriz_Acc, 'Xlim',[1 500], 'Ylim', [1 500], 'Xgrid', 'on', 'Ygrid', 'on', 'Zlim', [1 500], 'Zgrid', 'on','units','normalized');
handles.Matriz_Acc.XLabel.String='k';
handles.Matriz_Acc.YLabel.String='h';
handles.Matriz_Acc.ZLabel.String='number of votes';
%axes hist
cla(handles.histograma, 'reset');
set(handles.histograma, 'Xlim',[1 100], 'Ylim', [1 100], 'Xgrid', 'on', 'Ygrid', 'on','units','normalized');
handles.histograma.XLabel.String='radius';
handles.histograma.YLabel.String='number of votes';

[impath, user_canceled] = imgetfile;
if user_canceled
    msgbox(sprintf('The operation could be not completed'),'Error','Error');
    return;
end

%get image
Real_image=imread(impath);
%check format image
formatos= {'uint8', 'uint16', 'double', 'single'};
validateattributes(Real_image, formatos,{'nonempty','nonsparse','real'},mfilename,'Real_image',1);
%check image
N = ndims(Real_image);
if (isvector(Real_image) || N > 3)
    error(message('Imagen inválida'));
elseif (N == 3)
    if (size(Real_image,3) ~= 3)
        error(message('invalid image format'));
    end
end
A=rgb2gray(Real_image);
A = imbinarize(A);
[rows cols]=size(A);
rmax=round(max((rows)/2,(cols)/2));
axes(handles.Imagen_Original);
imshow(A);
hold off
axes(handles.Hough);
imshow(Real_image);
pause(1);

%typing radius values
value = get(handles.rango_radios, 'Value');
if value==true
    prompt={'Enter radius rank [rmin rmax]'};
    titulo='Method general Hough 3D';
    answer=inputdlg(prompt,titulo, [1 40]);
    if isempty(answer)
        msgbox(sprintf('The values entered are invalids'),'Error','Error');
        return;
    end
    rads = str2num(answer{:});
    rmin=rads(1);
    if (rads(2)<rmax)
        rmax=rads(2);
    end
else
    rmin=10;
end

%%%%%%% Hough 3D%%%%%%%%%%%%%%%
[Acc]=Hough3D(A,rows,cols,rmin,rmax);

%for histogram
Acchist=Acc;
Ac=Acc.^2;


%%%%%%%%%%%%%%% getting radio
max_num=max(Acc(:)); %value maxim of accumulator matrix
%checking region of custom search
value_region = get(handles.region,'Value');
if value_region == true
    prompt={'Define region of search of local maximums and difference between radius values: [width  height  (rmax-rmin)]'};
    titulo='Method general Hough 3D';
    answer=inputdlg(prompt,titulo, [1 45]);
    if isempty(answer)
        msgbox(sprintf('The values entered are invalids'),'Error','Error');
        return;
    end
    reg=str2num(answer{:});
    limx=reg(1);
    limy=reg(2);
    limr=reg(3);
else
    limx=round(0.1*cols);
    limy=round(0.1*rows);
    limr=15;
end


%Checking if knows the number of radius
value_cir = get(handles.no_circulos, 'Value');

%if knows circles in image
if value_cir==true
    prompt={'Enter the number of circles'};
    titulo='Method general Hough 3D';
    answer=inputdlg(prompt,titulo, [1 40]);
    if isempty(answer)
        msgbox(sprintf('The values entered are invalids'),'Error','Error');
        return;
    end
    n = str2num(answer{:}); %number of circles

   
    for i=1:max_num
        [K H R] = ind2sub(size(Acc),find(Acc >= max_num-i-1));
        [yk xk]=size(K);

        %if only wants and only exist a circumference
        if(n==1 && numel(H(H>0))==n)

            break;
        end
        %selecting a value j of H for comparing with a value k of H, this
        %is for search and delete repited peaks or noise
        for j=1:yk
            
            for k=yk:-1:1
                
                
                if(j==k)
                    continue;
                end
                
                %testing the same interval, if so, the k element is a
                %candidate to be deleted
                if ( H(j)>0 && H(k)>0 && K(j)>0 && K(k)>0 && H(j)+limx>H(k) && H(j)-limx<H(k) && K(j)+limy>K(k) && K(j)-limy<K(k) && R(j)+limr>R(k) && R(j)-limr<R(k) )

                    if (Acc(K(j),H(j),R(j))>=Acc(K(k),H(k),R(k)))
                                                
                        H(k)=0;
                        K(k)=0;
                        R(k)=0;
                        
                    else

                        H(j)=0;
                        K(j)=0;
                        R(j)=0;
                        
                    end

                end
                
            end
            
        end

        if(numel(H(H>0))==n)
            break;
        end

    end
    
    
else
    
    %entering threshold value for detecting peaks
    mnum=num2str(max_num);
    ms=num2str(round((0.5*max_num)));
    prompt={['Enter threshold for peaks (Advice: ' mnum '>threshold <= ' ms ')']};
    titulo='Method general Hough 3D';
    answer=inputdlg(prompt,titulo, [1 40]);
    if isempty(answer)
        msgbox(sprintf('The operation could be not completed'),'Error','Error');
        return;
    end
    umbral = str2num(answer{:});
    %getting maximums
    [K,H,R] = ind2sub(size(Acc),find(Acc >= max_num-umbral));
    [yk xk]=size(K);
    
     for j=1:yk
            
            for k=yk:-1:1
                
                %so as not to take the same element
                if(j==k)
                    continue;
                end
                
                %testing the same interval, if so, the k element is a
                %candidate to be deleted
                if ( H(j)>0 && H(k)>0 && K(j)>0 && K(k)>0 && H(j)+limx>H(k) && H(j)-limx<H(k) && K(j)+limy>K(k) && K(j)-limy<K(k) && R(j)+limr>R(k) && R(j)-limr<R(k) )

                    if (Acc(K(j),H(j),R(j))>=Acc(K(k),H(k),R(k)))
                                                
                        H(k)=0;
                        K(k)=0;
                        R(k)=0;
                        
                    else

                        H(j)=0;
                        K(j)=0;
                        R(j)=0;
                        
                    end

                end
                
            end
            
     end   
    
     
end
%%%%%%%%%%%%%%%
H=H(H>0);
K=K(R>0);
R=R(R>0);
Resultalt=[H K R];
C=Resultalt;


%Accumulator Matrix
if(handles.Mostrar==3)
    %plotting Accumulator Matrix in 3D view
    colorbar(handles.Matriz_Acc,'off');
    mesh(handles.Matriz_Acc,Acchist(:,:,R(1)));
    set(handles.Matriz_Acc, 'Xlim',[1 cols], 'Ylim', [1 rows], 'Xgrid', 'on', 'Ygrid', 'on', 'Zgrid', 'on','units','normalized','FontSize',12,'Ydir','reverse');
    title(handles.Matriz_Acc,['Accumulator Matrix en r= ', num2str(R(1))]);
    handles.Matriz_Acc.XLabel.String='h';
    handles.Matriz_Acc.YLabel.String='k';
    handles.Matriz_Acc.ZLabel.String='number of votes';
else
    %plotting Accumulator Matrix in 2D view
    slice(handles.Matriz_Acc,Acchist,[],[],R(1));
    colormap(handles.Matriz_Acc,jet);
    view(handles.Matriz_Acc,2)
    set(handles.Matriz_Acc, 'Xlim',[1 cols], 'Ylim', [1 rows], 'Xgrid', 'on', 'Ygrid', 'on','units','normalized','FontSize',12,'Ydir','reverse');
    title(handles.Matriz_Acc,['Accumulator Matrix en r= ', num2str(R(1))]);
    handles.Matriz_Acc.XLabel.String='h';
    handles.Matriz_Acc.YLabel.String='k';
    
    handles.Matriz_Acc;
    
    CM=colorbar(handles.Matriz_Acc);
    CM.Label.String ='number of votes';
    hold off
end

[yR xR]=size(R);


%plotting radius histogram
hist=zeros(rmax,1);
for i=rmin:rmax
    hr=max(Acchist(:,:,i));
    hist(i,1)=max(hr);
end

axes(handles.histograma);
bar(hist);
handles.histograma.XLabel.String='radius';
handles.histograma.YLabel.String='number of votes';
set(handles.histograma, 'Xlim',[rmin rmax], 'Xgrid', 'on', 'Ygrid', 'on','units','normalized');


%plotting found circle
axes(handles.Hough);
hold on;
plot(handles.Hough,H,K,'*','Color','g');
viscircles([H K],R,'Color','r');
hold off

%centers and radius
handles.tabla_radios.Data=C;

%handles variables
handles.Acchist=Acchist;
handles.rows=rows;
handles.cols=cols;
handles.yR=yR;
handles.R=R;
handles.Ac=Ac;
guidata(hObject, handles);



% --- Executes on button press in rango_radios.
function rango_radios_Callback(hObject, eventdata, handles)
% hObject    handle to rango_radios (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rango_radios


% --- Executes on button press in no_circulos.
function no_circulos_Callback(hObject, eventdata, handles)
% hObject    handle to no_circulos (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of no_circulos


% --- Executes on button press in slice.
function slice_Callback(hObject, eventdata, handles)
%this funcition allows to press the button for change slice
handles = guidata(hObject);
Acchist=handles.Acchist;
rows=handles.rows;
cols=handles.cols;
yR=handles.yR;
Ac=handles.Ac;
R=handles.R;
R=unique(R);
[yR yx]=size(R);
set(handles.Matriz_Acc, 'Xlim',[1 cols], 'Ylim', [1 rows], 'Xgrid', 'on', 'Ygrid', 'on', 'Zgrid', 'on','units','normalized');
handles.Matriz_Acc.XLabel.String='h';
handles.Matriz_Acc.YLabel.String='k';
handles.Matriz_Acc.ZLabel.String='number of votes';

if handles.counter<yR
    handles.counter=handles.counter+1;
    c=handles.counter;
    cla(handles.Matriz_Acc,'reset')
    colorbar(handles.Matriz_Acc,'off');
    %matrix Acc 3d
    if(handles.Mostrar==3)
        mesh(handles.Matriz_Acc,Acchist(:,:,R(c)));
        set(handles.Matriz_Acc, 'Xlim',[1 cols], 'Ylim', [1 rows], 'Xgrid', 'on', 'Ygrid', 'on', 'Zgrid', 'on','units','normalized','FontSize',12,'Ydir','reverse');
        title(handles.Matriz_Acc,['Accumulator Matrix en r= ', num2str(R(c))]);
        handles.Matriz_Acc.XLabel.String='h';
        handles.Matriz_Acc.YLabel.String='k';
        handles.Matriz_Acc.ZLabel.String='number of votes';
    else
        %matrix Acc 2d
        slice(handles.Matriz_Acc,Acchist,[],[],R(c));
        colormap(handles.Matriz_Acc,jet);
        view(handles.Matriz_Acc,2)
        set(handles.Matriz_Acc, 'Xlim',[1 cols], 'Ylim', [1 rows], 'Xgrid', 'on', 'Ygrid', 'on','units','normalized','FontSize',12,'Ydir','reverse');
        title(handles.Matriz_Acc,['Accumulator Matrix en r= ', num2str(R(c))]);
        handles.Matriz_Acc.XLabel.String='h';
        handles.Matriz_Acc.YLabel.String='k';
        handles.Matriz_Acc;
        CM=colorbar(handles.Matriz_Acc);
        CM.Label.String ='number of votes';
        hold off
    end
    
else
    %resetting counter
    handles.counter=0;
end


%%refresh
guidata(hObject, handles);



% --- Executes on selection change in Representacion_Matriz_Acc.
function Representacion_Matriz_Acc_Callback(hObject, eventdata, handles)
% hObject    handle to Representacion_Matriz_Acc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%Determine the selected data set.
str=get(hObject, 'String');
val=get(hObject, 'Value');
%Set current data to the selected data set.
%User selects peaks
switch str{val};
    case '2D view' 
        handles.Mostrar=2;
    case '3D view'
        handles.Mostrar=3;
end
%Save the handles structure
guidata(hObject,handles);
% Hints: contents = cellstr(get(hObject,'String')) returns Representacion_Matriz_Acc contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Representacion_Matriz_Acc


% --- Executes during object creation, after setting all properties.
function Representacion_Matriz_Acc_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Representacion_Matriz_Acc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
%return the initial color of the background
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in region.
function region_Callback(hObject, eventdata, handles)
% hObject    handle to region (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of region
