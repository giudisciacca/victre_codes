function slice_by_slice(V)

for i = 1:size(V,3)
   figure(1), imagesc(V(:,:,i)),axis image
   title(num2str(i)) 
   pause()
end



end