function s = sbx_file_info(fname)
% SBX_FILE_INFO  Frame geometry, channel layout and frame count of a
% Scanbox .sbx file, read from its info .mat (follows sbxread and
% sbx2tif_lily2025).
%
%   s = sbx_file_info('D:\data\250314_001_000')      % .sbx extension optional
%
% OUTPUT fields
%   path        - full path to the .sbx file
%   rows, cols  - frame size in px
%   nchan       - number of channels interleaved in the file
%   chs         - PMT channel numbers present (1 = green, 2 = red), in the
%                 order they are interleaved
%   frameBytes  - bytes per frame (all channels)
%   nFramesDisk - number of complete frames in the file

fname = char(fname);
if endsWith(fname, '.sbx', 'IgnoreCase', true)
    fname = fname(1:end-4);
end
if ~isfile([fname '.sbx'])
    error('Cannot find %s.sbx', fname);
end
if ~isfile([fname '.mat'])
    error('Cannot find Scanbox info file %s.mat', fname);
end
S = load([fname '.mat'], 'info');
info = S.info;
if ~isfield(info, 'sz')
    info.sz = [512 796];        % sbxread default for old files
end
if ~isfield(info, 'scanmode')
    info.scanmode = 1;          % unidirectional
end
rows = info.recordsPerBuffer;
if info.scanmode == 0
    rows = rows * 2;            % bidirectional
end
cols = info.sz(2);
if info.sz(1) ~= rows
    warning('%s: info.sz(1)=%d but recordsPerBuffer gives %d rows; using %d.', ...
        fname, info.sz(1), rows, rows);
end
if isfield(info, 'chan')                 % Pizzabox / newer Scanbox
    nchan = info.chan.nchan;
    if isfield(info.chan, 'save')
        chs = find(info.chan.save);
    else
        chs = 1:nchan;
    end
else
    switch info.channels
        case 1, nchan = 2; chs = [1 2];  % both PMTs
        case 2, nchan = 1; chs = 1;      % PMT 0
        case 3, nchan = 1; chs = 2;      % PMT 1
        otherwise, error('%s: unknown info.channels = %d', fname, info.channels);
    end
end
if numel(chs) ~= nchan
    error('%s: channel list %s does not match nchan = %d', fname, mat2str(chs), nchan);
end
d = dir([fname '.sbx']);
s.path = [fname '.sbx'];
s.rows = rows;
s.cols = cols;
s.nchan = nchan;
s.chs = chs(:)';
s.frameBytes = rows * cols * nchan * 2;
s.nFramesDisk = floor(d.bytes / s.frameBytes);
end
