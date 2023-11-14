function out = index_2darray(A,r,c)
idx = sub2ind(size(A), r, c);
out = A(idx);

end

