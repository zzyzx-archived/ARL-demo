function out = index_3darray(A,r,c)
out = zeros(length(r),size(A,3));
for i=1:length(r)
    out(i,:) = A(r(i),c(i),:);
end

end

