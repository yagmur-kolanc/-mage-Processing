%yağmur Kolancı

apply_gaussian_filter('flower.jpg', 2); %To run Step 5

%% 
%Step 1: Grayscale Conversion and Binary Imaging

%1.Grayscale Conversion: The image is read using imread and converted to grayscale 
% if it is in RGB format.

%2.Binary Imaging: The image is binarized (converted to black and white) using 
% the imbinarize function.

im = imread('flower.jpg');
if size(im, 3) == 3
    im = rgb2gray(im);
end

%To show the gray scale image
figure;
imshow(im);
title('Gray Scale Image');

binaryImage = imbinarize(im); %Matlab imbinarize function

%To show built-in binarize function output.
figure;
imshow(binaryImage);
title('Binarized Image');

%% 

% Step 2: Thresholding with Otsu's Method 

% 1.Image Reading and Grayscale Conversion: The image is read and converted to grayscale.

% 2.Threshold Value Determination: Threshold values from 50 to 250 are tested. 
% For each threshold value, the number of white and black pixels and their average values are calculated.

% 3.Finding Optimal Threshold Value: The threshold value with the maximum variance is found, 
% and the image is binarized again using this optimal threshold value.


im = imread('flower.jpg');
if size(im, 3) == 3
    im = rgb2gray(im);
end

levels = 50:50:250; % Possible threshold levels for more adequate thresholding
variances = zeros(size(levels)); % Array for variances

% Loop to find the optimal level among threshold levels
for i = 1:length(levels)
    level = levels(i);
    whites = 0; blacks = 0; mean_wh = 0; mean_bl = 0;
    
    % Loop for every pixel
    for m = 1:size(im, 1)
        for n = 1:size(im, 2)
            if im(m, n) > level
                whites = whites + 1;
                mean_wh = mean_wh + double(im(m, n));
            else
                blacks = blacks + 1;
                mean_bl = mean_bl + double(im(m, n));
            end
        end
    end
    
    if whites > 0
        mean_wh = mean_wh / whites;
    end
    if blacks > 0
        mean_bl = mean_bl / blacks;
    end
    
    variances(i) = whites * blacks * (mean_wh - mean_bl)^2;
end

% Find the level with the maximum variance
[~, max_index] = max(variances);
optimal_level = levels(max_index);

% Binarize the image again with the optimal level
binaryImage_optimal = imbinarize(im, optimal_level / 255);

% Display the results
figure;
subplot(1, 2, 1);
imshow(im);
title('Original Grayscale Image');
subplot(1, 2, 2);
imshow(binaryImage_optimal);
title(['Binarized Image with Optimal Level: ', num2str(optimal_level)]);



%% 

%Step 3 TODO: Roberts Filter
im = imread('flower.jpg');% Read the 'flower.jpg' file and assign it to 'im'

if size(im, 3) == 3 % If the image is RGB (colored) (3 channels)
    im = rgb2gray(im); % Convert the image to grayscale
end

%fspecial functions
meanFilter = fspecial('average', [3 3]);% Create an average filter of size 3x3
gaussFilter = fspecial('gaussian', [3 3], 0.5); % Create a Gaussian filter of size 3x3 with a sigma value of 0.5 
prewittFilter = fspecial('prewitt');%% Create a Prewitt edge detection filter
sobelFilter = fspecial('sobel'); % Create a Sobel edge detection filter
% robertsFilterX = [1 0; 0 -1];% Roberts edge detection filter in the X direction (commented out)
% robertsFilterY = [0 1; -1 0];% Roberts edge detection filter in the Y direction (commented out)
logFilter = fspecial('log', [3 3], 0.5);%Create a Laplacian of Gaussian (LoG) filter of size 3x3 with a sigma value of 0.5
    
meanFiltered = imfilter(im, meanFilter); % Apply the average filter to the image 
gaussFiltered = imfilter(im, gaussFilter); % Apply the Gaussian filter to the image
prewittFiltered = imfilter(im, prewittFilter); % Apply the Prewitt filter to the image
sobelFiltered = imfilter(im, sobelFilter);% Apply the Sobel filter to the image
% robertsFilteredX = imfilter(im, robertsFilterX); % Apply the Roberts filter in the X direction (commented out)
% robertsFilteredY = imfilter(im, robertsFilterY); % Apply the Roberts filter in the Y direction (commented out)
% robertsFiltered = sqrt(robertsFilteredX.^2 + robertsFilteredY.^2); %Combine the results of the X and Y direction Roberts filters (commented out)
logFiltered = imfilter(im, logFilter); % Apply the Laplacian of Gaussian filter to the image
    
    
figure;% Create a new figure
subplot(3, 3, 1); imshow(im); title('Original Image'); % Display the original image
subplot(3, 3, 2); imshow(meanFiltered); title('Mean Filter'); % Display the image filtered with the average filter
subplot(3, 3, 3); imshow(gaussFiltered); title('Gaussian Filter'); title('Gaussian Filter'); % Display the image filtered with the Gaussian filter
subplot(3, 3, 4); imshow(prewittFiltered); title('Prewitt Filter'); title('Prewitt Filter'); % Display the image filtered with the Prewitt filter
subplot(3, 3, 5); imshow(sobelFiltered); title('Sobel Filter'); % Display the image filtered with the Sobel filter
% subplot(3, 3, 6); imshow(robertsFiltered); title('Roberts Filter'); % Display the image filtered with the Roberts filter (commented out)
subplot(3, 3, 7); imshow(logFiltered); title('Laplacian of Gaussian');% Display the image filtered with the Laplacian of Gaussian filter

%% 

%Step 4

my_filter('flower.jpg');% Call the custom filter function with 'flower.jpg'

function my_filter(imagePath)
    im = imread(imagePath);% Read the image from the given path
    if size(im, 3) == 3 % If the image is RGB (colored)
        im = rgb2gray(im);% Convert the image to grayscale
    end
    
    meanFilteredRows = filter_rows(im, @mean_filter);% Apply row-wise mean filter
    meanFiltered = filter_columns(meanFilteredRows, @mean_filter); % Apply column-wise mean filter
    
    medianFilteredRows = filter_rows(im, @median_filter);% Apply row-wise median filter
    medianFiltered = filter_columns(medianFilteredRows, @median_filter); % Apply column-wise median filter
    
    % Display the results
    figure;
    subplot(1, 3, 1); imshow(im); title('Original Image');% Display the original image
    subplot(1, 3, 2); imshow(meanFiltered); title('Mean Filter');% Display the image filtered with the mean filter
    subplot(1, 3, 3); imshow(medianFiltered); title('Median Filter');% Display the image filtered with the median filter
end

function output = filter_rows(im, filter_func)
    [rows, cols] = size(im);% Get the size of the image
    output = zeros(size(im));% Initialize the output image
    for r = 1:rows % Loop through each row
        output(r, :) = filter_func(im(r, :));% Apply the filter function to the row
    end
end

function output = filter_columns(im, filter_func)
    [rows, cols] = size(im);  % Get the size of the image
    output = zeros(size(im)); % Initialize the output image
    for c = 1:cols % Loop through each column
        output(:, c) = filter_func(im(:, c)); % Apply the filter function to the column
    end
end

%mean filter
function result = mean_filter(vec)
    vec = double(vec);% Convert the vector to double
    n = length(vec);% Get the length of the vector
    result = zeros(size(vec)); % Initialize the result vector
    for i = 1:n % Loop through each element in the vector
        if i == 1
            result(i) = mean(vec(i:i+1));% Apply mean filter for the first element
        elseif i == n
            result(i) = mean(vec(i-1:i));% Apply mean filter for the last element
        else
            result(i) = mean(vec(i-1:i+1));% Apply mean filter for all other elements
        
        end
    end
end

%median filter
function result = median_filter(vec)
    vec = double(vec); % Convert the vector to double
    n = length(vec);% Get the length of the vector
    result = zeros(size(vec));  % Initialize the result vector
    for i = 1:n % Loop through each element in the vector
        if i == 1
            result(i) = median(vec(i:i+1));% Apply median filter for the first element
        elseif i == n
            result(i) = median(vec(i-1:i));% Apply median filter for the last element
        else
            result(i) = median(vec(i-1:i+1)); % Apply median filter for all other elements
        end
    end
end

%% 

%Step 5

function G = gauss2(sigma)
    %Defining filter size
    [x, y] = meshgrid(-ceil(3 * sigma):ceil(3 * sigma), -ceil(3 * sigma):ceil(3 * sigma));% Create a grid of coordinates
    G = exp(-(x.^2 + y.^2) / (2 * sigma^2));% Calculate the Gaussian function
    G = G / sum(G(:));  % Normalize the filter so that the sum of all elements is 1
end

function apply_gaussian_filter(imagePath, sigma)
    im = imread(imagePath); % Read the image from the given path
    if size(im, 3) == 3 % Read the image from the given path
        im = rgb2gray(im);% Convert the image to grayscale
    end
    
    im = double(im); %convert the image to double
    %this is required for filtering operations
    
    G = gauss2(sigma); %Gaussian Filter
                       % Create the Gaussian filter with the specified sigma value
    
    gaussianFiltered = imfilter(im, G, 'replicate');% Apply the Gaussian filter to the image
    
    gaussianFiltered = uint8(gaussianFiltered); %converting double to 
    %uint8 type for displaying.
    
    figure;
    subplot(1, 2, 1); imshow(uint8(im)); title('Original Image');
    subplot(1, 2, 2); imshow(gaussianFiltered); title('Gaussian Filtered Image');
end

