% Load images in folder for labeling
% Cathey Wang

imgDir = '../Cropped/';

ls = dir(imgDir);
ls = ls(3:end);
FF = length(ls);

imgFiles = cell(FF,1);
for i = 1:FF
    fn = ls(i).name;
    imgFiles{i} = [imgDir, fn];
end

for i = 43:FF
    im = imread(imgFiles{i});
    imagesc(im)
    title(imgFiles{i})
    pause;
end