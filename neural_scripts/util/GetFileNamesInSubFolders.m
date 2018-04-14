function [ fileNames ] = GetFileNamesInSubFolders( rootDir, ext )
% Given a directory 'rootDir', return all files' paths in subfolders ending in
% 'ext'

if nargin < 2
    ext = '';
end


listing = dir(rootDir);
fileNames = cell(5000000,1);
currIndx = 1;

for i = 3:numel(listing)
    if listing(i).isdir
        newFileNames = GetFileNamesInSubFolders([rootDir, '/', listing(i).name], ext);
        fileNames(currIndx : currIndx+numel(newFileNames)-1) = newFileNames;
        currIndx = currIndx + numel(newFileNames);
    elseif endsWith(listing(i).name, ext)
        fileNames{currIndx} = [rootDir, '/', listing(i).name];
        currIndx = currIndx + 1;
    end    
end

% Remove excess space
fileNames(currIndx:end) = [];

end

