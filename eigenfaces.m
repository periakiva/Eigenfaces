%%Peri Akiva
%Project 4, Part 2

clc
clear

input_dir = 'path\to\train\';
testdir='D:\PERI\Documents\MATLAB\test\'
image_dims = [200, 180];
classes=[0;1;2;3;4;5;6;7;8;9];
 

num_images = 170;
images = [];
imageclasses=[];
disp(num_images);
counter = 1;
c = cell(170,1);
c{1}={''};
file=cell(170,1);
file(1)={''};
paths = fullfile(testdir,{'0','1','2','3','4','5','6','7','8','9'});
for i = classes(1):classes(10)
    
    filenames = dir(fullfile(input_dir,int2str(i),'\*.jpg'));
    for n = 1:17
     filename = fullfile(input_dir,int2str(i),filenames(n).name);
     j=cellstr(filename);
     k=cellstr(filenames(n).name);
     file{counter}=[k];
     img = imread(filename);
     img = rgb2gray(img);
     img = im2double(img);
     c{counter}=[j];
     images(:, counter) = img(:);
     imageclasses(:,counter)=i;
     counter=counter+1;
    end
    
end


mean_face = mean(images, 2);
shifted_images = images - repmat(mean_face, 1, num_images);


[u, s, v] = pca(images');
num_eigenfaces = 20;
u = u(:, 1:num_eigenfaces);

features = u' * shifted_images;
testlabel=[];
cc=1;
testimages=[];
featuresvector=[];
predicition=[];
%testing
for i = classes(1):classes(10)
    filenames1 = dir(fullfile(testdir,int2str(i),'\*.jpg'));
    for n = 1:3
     filename1 = fullfile(testdir,int2str(i),filenames1(n).name);
     
     img = imread(filename1);
     img = rgb2gray(img);
     img = im2double(img);
     figure(1)
     imshow(img)
   % if n == 1
      %  images = zeros(prod(image_dims), num_images);
  %  end
     
     input_image(:, cc) = img(:);
     testlabel(:,cc)=i;
     
     feature_vec = u' * (img(:) - mean_face);

     similarity = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);

     % find the image with the highest similarity
     [match, matchidx] = max(similarity);
     a = reshape(images(:,matchidx), image_dims);
     predicition(:,cc)=imageclasses(matchidx);
     % display the result
     %figure
     %A=cell2struct(file,'name',1)
      if 3%n=0
         figure
         imshow([img reshape(images(:,matchidx), image_dims)]);
         title(sprintf('matches %f, score %0.2f', 1-match));
      end
    
     cc=cc+1;
     
    end
    
end

confusionMatrix=confusionmat(testlabel,predicition)

%show mean face
a = reshape(mean_face, image_dims);
figure
imshow(a)


% display the eigenfaces
figure;
for n = 1:num_eigenfaces
    subplot(2, ceil(num_eigenfaces/4), n);
    evector = reshape(u(:,n), image_dims);
    imagesc(evector);
    colormap(gray);
end

