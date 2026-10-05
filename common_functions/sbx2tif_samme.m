function outFiles = sbx2tif_samme(sbxFile, varargin)
% SBX2TIF  Convert a Scanbox .sbx recording to uint16 BigTIFF, one file
% per channel (<name>_ch1.tif, <name>_ch2.tif)
%
%   outFiles = sbx2tif('D:\data\250314_001_000')          % .sbx optional
%   outFiles = sbx2tif(f, 'channels', 1, 'crop', [1 80 Inf Inf])
%
% Reads the .sbx in chunks (fast, low memory; no frame-by-frame growing
% arrays), inverts the values as sbxread does, and writes uncompressed
% BigTIFF so files > 4 GB work.
%
% OPTIONAL NAME-VALUE INPUTS (defaults in brackets)
%   'channels'   ([] = all recorded) PMT channels to write (1 = green, 2 = red)
%   'crop'       ([1 80 Inf Inf]) [row1 col1 row2 col2] of the raw frame.
%                Default drops the first 79 columns (bidirectional scan
%                turnaround band), same as sbx_motion_correct and the old
%                pipeline; [1 1 Inf Inf] keeps the full raw frame. Saved,
%                with the raw frame size, to <name>_tifinfo.mat.
%   'frames'     ([1 Inf]) [first last] frames to write, 1-based
%   'outDir'     ([] = next to the .sbx) folder for the tifs
%   'chunkSize'  (500) frames read per chunk
%   'overwrite'  (false) replace existing tifs
%   'verbose'    (true) print progress
%
% OUTPUT
%   outFiles - cell array of the tif paths written (or already present)

ip = inputParser;
ip.addParameter('channels', []);
ip.addParameter('crop', [1 80 Inf Inf]);
ip.addParameter('frames', [1 Inf]);
ip.addParameter('outDir', []);
ip.addParameter('chunkSize', 500);
ip.addParameter('overwrite', false);
ip.addParameter('verbose', true);
ip.parse(varargin{:});
p = ip.Results;

base = regexprep(char(sbxFile), '\.sbx$', '', 'ignorecase');
s = sbxInfo(base);
chans = p.channels;
if isempty(chans)
    chans = s.chs;
end
if ~all(ismember(chans, s.chs))
    error('%s: channel(s) %s not recorded (channels: %s)', base, mat2str(chans), mat2str(s.chs));
end

% crop and frame range, clipped to the file
c = p.crop(:)';
rc = [max(c(1:2), 1), min(c(3:4), [s.rows s.cols])];
fr = [max(p.frames(1), 1), min(p.frames(2), s.nFrames)];
if fr(2) < fr(1)
    error('%s: no frames in range %s (file has %d)', base, mat2str(p.frames), s.nFrames);
end
h = rc(3) - rc(1) + 1;
w = rc(4) - rc(2) + 1;

% output files
[d, n] = fileparts(base);
if ~isempty(p.outDir)
    d = p.outDir;
    if ~isfolder(d)
        mkdir(d);
    end
end
outFiles = arrayfun(@(ch) fullfile(d, sprintf('%s_ch%d.tif', n, ch)), chans, 'UniformOutput', false);
todo = p.overwrite | ~isfile(outFiles);
if ~any(todo)
    if p.verbose, fprintf('sbx2tif: %s already converted, skipping\n', n); end
    return
end
chans = chans(todo);
todoIdx = find(todo);
tw = cell(1, numel(chans));
for k = 1:numel(chans)
    tw{k} = Tiff(outFiles{todoIdx(k)}, 'w8');   % BigTIFF
end
closeAll = onCleanup(@() cellfun(@close, tw));
tags = struct('ImageLength', h, 'ImageWidth', w, ...
    'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
    'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
    'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
    'Compression', Tiff.Compression.None, 'RowsPerStrip', h);

% read in chunks, write every channel
fid = fopen(s.path, 'r');
closeFile = onCleanup(@() fclose(fid));
fseek(fid, (fr(1) - 1) * s.frameBytes, 'bof');
nTotal = fr(2) - fr(1) + 1;
ci = arrayfun(@(ch) find(s.chs == ch), chans);   % position of each channel on disk
written = 0;
tStart = tic;
while written < nTotal
    m = min(p.chunkSize, nTotal - written);
    x = fread(fid, s.frameBytes / 2 * m, 'uint16=>uint16');
    if numel(x) ~= s.frameBytes / 2 * m
        error('%s: short read at frame %d', s.path, fr(1) + written);
    end
    % on disk: [chan col row frame], values inverted (as in sbxread)
    x = permute(reshape(x, s.nchan, s.cols, s.rows, m), [3 2 4 1]);   % [row col frame chan]
    x = intmax('uint16') - x(rc(1):rc(3), rc(2):rc(4), :, ci);
    for k = 1:numel(chans)
        for f = 1:m
            if written + f > 1
                tw{k}.writeDirectory();
            end
            tw{k}.setTag(tags);
            tw{k}.write(x(:, :, f, k));
        end
    end
    written = written + m;
    if p.verbose
        fprintf('sbx2tif %s: %d/%d frames (%.0f s)\n', n, written, nTotal, toc(tStart));
    end
end
clear closeAll closeFile
crop = struct('rawSize', [s.rows s.cols], 'raw', rc, 'outSize', [h w]); 
frames = fr; 
save(fullfile(d, [n '_tifinfo.mat']), 'crop', 'frames', 'outFiles');
end


function s = sbxInfo(base)
% Frame geometry, channels and frame count (follows sbxread).
if ~isfile([base '.sbx']) || ~isfile([base '.mat'])
    error('Need both %s.sbx and its Scanbox info file %s.mat', base, base);
end
S = load([base '.mat'], 'info');
info = S.info;
if ~isfield(info, 'sz'), info.sz = [512 796]; end
if ~isfield(info, 'scanmode'), info.scanmode = 1; end
s.rows = info.recordsPerBuffer * (1 + (info.scanmode == 0));   % bidirectional: x2
s.cols = info.sz(2);
if isfield(info, 'chan')                      % Pizzabox / newer Scanbox
    s.nchan = info.chan.nchan;
    s.chs = 1:s.nchan;
    if isfield(info.chan, 'save')
        s.chs = find(info.chan.save);
    end
else
    switch info.channels
        case 1, s.nchan = 2; s.chs = [1 2];   % both PMTs
        case 2, s.nchan = 1; s.chs = 1;       % PMT 0 (green)
        case 3, s.nchan = 1; s.chs = 2;       % PMT 1 (red)
        otherwise, error('%s: unknown info.channels = %d', base, info.channels);
    end
end
s.path = [base '.sbx'];
s.frameBytes = s.rows * s.cols * s.nchan * 2;
d = dir(s.path);
s.nFrames = floor(d.bytes / s.frameBytes);
end
