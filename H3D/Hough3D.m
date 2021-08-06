%program of circle detection using Hough transform with matrix of dimension
%three.
%A: m-by-n matrix
%rows: integer number
%cols: integer number
%rmin: integer number
%rmax integer number
function [Acc]=Hough3D(A,rows,cols,rmin,rmax)
Acc=zeros(rows,cols,rmax);
for r=rmin:1:rmax
    for x=1:cols
        for y=1:rows
            if(A(y,x)==0)
                for ang=0:360
                    t=(ang*pi)/180;
                    x0=round(x-r*cos(t));
                    y0=round(y-r*sin(t));
                    if(x0<cols && x0>0 && y0<rows && y0>0)
                        Acc(y0,x0,r)=Acc(y0,x0,r)+1; %votes
                    end
                end
            end
        end
    end
end
end