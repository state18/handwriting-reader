function word_ascii2mat(filePath)
% Parses word labels from text file to .mat format

fid = fopen(filePath);
% Ignore first 18 lines.
for i= 1:19
    tline = fgetl(fid);
end

% Initialize struct to be saved in .mat file.
words = struct('ID', '', 'Quality', false, 'Graylevel', 0, 'Word', '');
words = repmat(words, 115500, 1);
sIndex = 1;

while ischar(tline)
    % Ignore blank lines.
    
    % Keep image ID, seg quality, graylevel, and transcription
    % Separate by spaces.
    pieces = strsplit(tline, ' ');
    words(sIndex).ID = pieces{1};
    
    qual = pieces{2};
    switch qual
        case 'ok'
            words(sIndex).Quality = 1;
        case 'err'
            words(sIndex).Quality = 0;
        otherwise
            error('Unrecognized quality string!');
    end
    
    words(sIndex).Graylevel = pieces{3};
    words(sIndex).Word = pieces{9};
    
    sIndex = sIndex + 1;
    
    tline = fgetl(fid);
end

% Trim final output
words(sIndex:end) = [];
mkdir('mat');
save('mat\iams_words.mat', 'words');
fclose(fid);
end
