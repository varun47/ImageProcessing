% Filename : ehsw.m

% Fast and efficient histogram based sliding window

% Yichen Wei, Litian Tao: Efficient histogram-based sliding window. CVPR 2010.

% Author : Varun Santhaseelan
% Date : 11/2/2012

% template - Template to be matched in the test image.
% im       - Test image where the template is to be matched
% F_out    - A matrix as the size of im that contains the matching scores at
%            each location

function F_out = ehsw(template, im)

% For the histogram
nbins           = 50;   % Number of bins
low_limit       = 0;    % Lowest value possible in the data to be matched.
high_limit      = 1;    % Highest value possible in the data to be matched.
interval        = (high_limit - low_limit) / (nbins-1);
bins            = low_limit:interval:high_limit;

[rows cols] = size(im);
[rows_t cols_t] = size(template);

% Output image is F.
F_out = 99999 * ones(rows,cols);

% Find the histogram of the template.
m = histc(template(:),bins);
m = m';

% Initialization for template matching
c = zeros(cols,nbins);
h = zeros(1,nbins);
d = zeros(1,nbins);
for j = 1:cols_t
    tmp = im(1:rows_t,j);
    c(j,:) = histc(tmp(:),bins);
    h(1,:) = h(1,:) + c(j,:);
end
for j = cols_t+1:cols
    tmp = im(1:rows_t,j);
    c(j,:) = histc(tmp(:),bins);
end

% Find the matching score for the first location.
F = 0;
for b=1:nbins
    d(1,b) = fb(h(1,b),m(1,b));
    F = F + d(1,b);
end
F_out(1,1) = F; 

% Continue finding the scores for the whole image.
for i = 1:rows-rows_t
    for j = 1:cols-cols_t
        dh = c(j+cols_t,:) - c(j,:);
        for b = 1:nbins
            if dh(1,b) ~= 0
                h(1,b) = h(1,b) + dh(1,b);
                tmp = fb(h(1,b),m(1,b));
                F = F + tmp - d(1,b);
                d(1,b) = tmp;
            end
        end
        F_out(i,j+1) = F;
    end
    for j = 1:cols
        % Replace im(i,j) with im(i+r,j) in the column histograms.
        indx = floor(im(i,j)/interval) + 1;
        c(j,indx) = c(j,indx) - 1;
        indx = floor(im(i+rows_t,j)/interval) + 1;
        c(j,indx) = c(j,indx) + 1;
    end
    h = zeros(1,nbins);
    for j = 1:cols_t
        h(1,:) = h(1,:) + c(j,:);
    end
    
    % Find the matching score for the first location.
    F = 0;
    for b=1:nbins
        d(1,b) = fb(h(1,b),m(1,b));
        F = F + d(1,b);
    end
    F_out(i,1) = F; 

end 
end

% Function to find Euclidean distance
function distance = fb(hb, mb)
    distance = (hb - mb)^2;
end
