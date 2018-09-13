function w = gausswin_loc(L)
%GAUSSWIN Gaussian window.
% directly taken from matlab


a = 2.5;
L = double(L); % data type of L is checked in check_order
N = L-1;
n = (0:N)'-N/2;
w = exp(-(1/2)*(a*n/(N/2)).^2);

end