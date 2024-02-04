%% main
% Load the pre-trained VGGFace model
net = vgg16(); 

% Read and preprocess the input image

%classImage = imread('./dataset2/IMG_9263.jpg');
% classImage = imread('./dataset2/class2.jpg');
% classImage = imread('./dataset1/first.jpg');
classImage =imread('./d2/class1.jpg');
% classImage = imread('adipclass.jpeg');

% Detect faces in the input image
boundingBoxObject = detectFaces(classImage);

% Initialize a flag to check if any face was recognized in image
recognized = false;

%process referenceimages of our dataset
referenceImages = preprocessReferenceImages();
% Create or open the attendance file
attendanceFile = fopen('attendance.txt', 'w');
% Create a cell array to store names
maxFaces=30;
presentNames = {};
count_present_students = 1;

% Process each detected face
for i = 1:size(boundingBoxObject, 1)
    % Extract the face region using imcrop
    faceBoundingBox = boundingBoxObject(i, :);
    face = imcrop(classImage, faceBoundingBox);
    
    % filename = sprintf('face%d.jpg',i);

    % % Save the cropped face image
    % imwrite(face, filename);

    %process the recognized face
    recognizedFace = preprocessImage(face);

    %find the person recognized 
    [recognizedPerson, maxSimilarity] = RecognizePerson(net, recognizedFace, referenceImages);
  
    % Set a similarity threshold to determine if it's a match
    similarityThreshold = 0.83;
   
    % Check if the maximum similarity is above the threshold
    if maxSimilarity >= similarityThreshold
        disp(['Face ', num2str(i), ' Recognized as ' recognizedPerson]);
        disp(maxSimilarity);
        recognized = true;
        % Write attendance to the text file
        %fprintf(attendanceFile, '%s Present\n', recognizedPerson);
         % Append the recognized name to the cell array
        presentNames{count_present_students} = recognizedPerson;
        count_present_students = count_present_students+1;
    
    else
        disp(['Face ', num2str(i), ' Not recognized as any person']);
    end
end
if ~recognized
    disp('No recognized faces in the image');
end

my_class_names = {'Safi','Sarah','Shayan', 'Anoosha', 'Areeb', 'Zainab Raza', 'Siqandar', 'Youshay', 'Zainab Haider', 'Mustufa', 'Murtuza'};
% Create or open the attendance file
attendanceFile = fopen('attendance.txt', 'w');


%Iterate over the names array
for i = 1:numel(my_class_names)
    % Check if the name exists in presentNames
    if ~ismember(my_class_names{i}, presentNames)
        % Write the name and 'Absent' to the text file
        fprintf(attendanceFile, '%s Absent\n', my_class_names{i});
    else
         fprintf(attendanceFile, '%s Present\n', my_class_names{i});
    end
    
end

% Close the attendance file
fclose(attendanceFile);

%% process reference images
function processedReferenceImages = preprocessReferenceImages()

    % Define a cell array of reference image file paths
    referenceImagePaths = {
        % './dataset2/test1.jpg',
        % './dataset2/test2.jpg',
        % './dataset2/test3.jpg',
         % './dataset2/test4.jpg',
        
        % 'test01.jpg'
        './d2/new1.jpg'%areeb
        './d2/n1.jpg'%shayan
        './d2/n2.jpg'%youshay
        './d2/new3.jpg'%safi
      
        './d2/new5.jpg' %zainab
        './d2/new7.jpg'%bushra
        './d2/new8.jpg'%sarah
        './d2/new9.jpg'%mustufa
        './d2/new10.jpg'%murtuza
        './d2/new11.jpg'%zainab raza

    };

    processedReferenceImages = cell(1, numel(referenceImagePaths));

    for i = 1:numel(referenceImagePaths)
        % Load the reference image
        referenceImage = imread(referenceImagePaths{i});

        %process the reference image
        referenceImage = preprocessImage(referenceImage);

        % Store the processed reference image in the cell array
        processedReferenceImages{i} = referenceImage;
    end
end

%% preprocess image

function resultImage = preprocessImage(image)   
    inputSize = [224, 224];
    meanImageNetRGB = [123.68, 116.78, 103.94];

    %resize image to inputSize of VGG  model
    image = imresize(image, inputSize);
    
    %convert image to single precision
    recognizedFace = single(image);
    
    % Subtract the mean RGB value from each channel 
    % meanImageNetRGB value of VGG model
   recognizedFace(:, :, 1) = recognizedFace(:, :, 1) - meanImageNetRGB(1);
   recognizedFace(:, :, 2) = recognizedFace(:, :, 2) - meanImageNetRGB(2);
   recognizedFace(:, :, 3) = recognizedFace(:, :, 3) - meanImageNetRGB(3);
    
    % Rearrange the dimensions of the image to match the expected format
    % [height, width, channels] of vgg 
    resultImage = permute(recognizedFace, [2, 1, 3]);
    
end

%% detect faces

function bbox = detectFaces(currentImage)

    % Load the cascade object detector
    faceDetection = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
    faceDetection.MergeThreshold = 10;
    faceDetection.MinSize = [100, 100];
    
    % Convert the image to grayscale for face detection
    % grayImage = rgb2gray(currentImage);

    % Detect faces
    % bbox = step(faceDetection, currentImage);
   bbox = faceDetection(currentImage); 

    % Define annotation properties
    boxColor = [0, 255, 0];

    % Annotate and display the image

    labeledImage = currentImage;  % Create a copy of the original image

    for i = 1:size(bbox, 1)
        % Draw a bounding box around the face with label

        label = ['Face ' num2str(i)];
        labeledImage = insertObjectAnnotation(labeledImage, 'rectangle', bbox(i, :), label, 'Color', boxColor, 'FontSize', 14, 'LineWidth',5);
    end

    figure, imshow(labeledImage);
end


%% face recognition

function [recognizedPerson, maxSimilarity] = RecognizePerson(net, recognizedFace, referenceImages)

    % Initialize variables for similarity and recognized person
    maxSimilarity = 0.8;
    recognizedPerson = 'Unknown';

    %extract features using VGG 16 
    embeddingRecognized = activations(net, recognizedFace, 'fc7', 'OutputAs', 'rows');


    % Calculate similarity with reference faces and find the maximum
    for j = 1:numel(referenceImages)

        embeddingReference = activations(net, referenceImages{j}, 'fc7', 'OutputAs', 'rows');

        %find cosine similarity
        similarity = dot(embeddingRecognized, embeddingReference) / (norm(embeddingRecognized) * norm(embeddingReference));
       
        if similarity >= maxSimilarity
            maxSimilarity = similarity;
   
            % Customize recognized person's name based on index
             if(j==1)
                recognizedPerson ='Areeb';
            elseif(j==2)
                recognizedPerson = 'Shayan';
             elseif(j==3)
                 recognizedPerson = 'Youshay';
            elseif(j==4)
                  recognizedPerson = 'Safi';
            elseif(j==5)
                   recognizedPerson = 'Zainab';
            elseif(j==6)
                   recognizedPerson = 'Bushra';
            elseif(j==7)
                   recognizedPerson = 'Sarah';
            elseif(j==8)
                   recognizedPerson = 'Mustufa';
           elseif(j==9)
                   recognizedPerson = 'Murtuza';
            elseif(j==10)
                   recognizedPerson = 'Zainab Raza';
            else
                recognizedPerson = ['Person ' num2str(j)];
            end
        end
    end
end