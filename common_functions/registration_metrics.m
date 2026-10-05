function M = registration_metrics(mov, varargin)
% REGISTRATION_METRICS  Suite2p-style PC-based motion-correction quality metrics.
%
% Implements "Registration metrics" from Stringer & Pachitariu,
% "Extracting large-scale neural activity with Suite2p" (bioRxiv 2026):
%   1. subsample 2000-5000 registered frames, crop to valid region
%   2. PCA -> top nPC components
%   3. for each PC: mean of nTop frames with largest / smallest projection
%   4. register bottom-mean to top-mean (rigid, then non-rigid on blocks)
%   5. report offset lengths: rigid, mean NR, max NR, mean(rigid+NR)
% Plus two more direct checks, on every frame of the (cropped) movie:
%   6. frame-to-mean correlation (as in CaImAn/NoRMCorre): Pearson
%      correlation of each frame with the mean image. Higher and flatter
%      after correction is better; remaining dips = residual motion.
%   7. kymographs: one horizontal and one vertical line through the mean
%      image (auto-picked where the high-pass mean image has the most edges,
%      or set with kymoRow / kymoCol). The band of kymoWidth px around each
%      line is averaged into one profile per frame, and profiles are averaged
%      over kymoBin frames. Straight stripes over time = stable; stripes that
%      bend together = residual motion; stripes that fade without moving =
%      likely z-motion.
% If 'rawMov' (the movie before correction, same pixels) is given, 6 and 7
% are also computed for it, on the same crop and lines, for before/after
% (and 1-5 too if 'rawPC' is true).
%
% Movies too large for memory: pass the tif path instead of the array. The
% movie is then read in passes and never held whole:
%   - sampling pass: only the 2000-5000 PCA sample frames are read (kept in
%     their native class, e.g. uint16: ~3.4 GB for 5000 frames of 493x692).
%     Their mean is used as the mean image for 6 and 7 (evenly spaced
%     frames, so it matches the full mean closely). PCA uses the
%     frames x frames Gram matrix, built over blocks of pixels.
%   - streaming pass: chunk frames at a time for 6 and 7.
% Counting the frames of a tif walks its page headers once (fast, no image
% data). An in-memory array is processed the same way.
%
%   M = registration_metrics(mov)
%   M = registration_metrics(mov, 'nPC',30, 'nTop',300, 'blockSize',[128 128], ...
%                            'yrange',[], 'xrange',[], 'smoothSigma',1.15, ...
%                            'maxShiftFrac',0.1, 'maxShiftNR',5, 'plot',true, ...
%                            'rawMov',[], 'rawPC',false, 'kymoRow',[], 'kymoCol',[], ...
%                            'kymoWidth',5, 'kymoBin',5, 'hpSigma',10, ...
%                            'maxFrames',Inf, 'chunk',200)
%
%   mov        registered movie: Ly x Lx x T array, a tif path (e.g.
%              ConcatenatedMovie_MC.tif), or a struct with field 'path' and
%              optional 'rows', 'cols' (only those pixels are used)
%   rawMov     movie before correction ([] = skip), same forms as mov. It
%              must cover the same pixels as mov. For sbx_motion_correct
%              output and a raw tif exported with the same 'crop', use
%              struct('path', rawTif, 'rows', S.outRows, 'cols', S.outCols)
%              with S from the _shifts.mat. Its frame count may differ from
%              mov (e.g. first frame of each segment not dropped).
%   rawPC      also run the PC metric (1-5) on rawMov
%   kymoRow    row (in the yrange/xrange crop) of the horizontal kymograph
%              line ([] = auto)
%   kymoCol    column (in the crop) of the vertical line ([] = auto)
%   kymoWidth  band width (px) averaged across the line
%   kymoBin    frames averaged per kymograph time bin
%   hpSigma    Gaussian sigma (px) of the high-pass filter used to auto-pick
%              the lines
%   maxFrames  use only the first maxFrames frames of each movie
%   chunk      frames per read in the streaming pass
%
% Output struct M:
%   regDX     nPC x 4  [rigid, meanNR, maxNR, mean(rigid+NR)] offset lengths (px)
%   tPC       nFrames x nPC   PC time courses
%   topMean   ly x lx x nPC   mean of top frames per PC (cropped)
%   botMean   ly x lx x nPC   mean of bottom frames per PC
%   frameIdx  indices of sampled frames
%   meanImg   mean image of the cropped movie (mean of the sampled frames)
%   frameCorr T x 1  correlation of each frame with meanImg
%   movSize   [Ly Lx T] of mov (after maxFrames)
%   frameCorrRaw, meanImgRaw, rawSize   same for rawMov ([] if not given)
%   regDXRaw, tPCRaw, topMeanRaw, botMeanRaw, frameIdxRaw   PC metric of
%             rawMov ([] unless rawPC)
%   kymo      struct: row, col (line positions in the crop), width, bin,
%             t (center frame of each time bin), H (lx x nBins, along the
%             horizontal line), V (ly x nBins, along the vertical line),
%             tRaw, Hraw, Vraw (same for rawMov, [] if not given)

p = inputParser;
p.addParameter('nPC', 30);
p.addParameter('nTop', 300);
p.addParameter('blockSize', [128 128]);
p.addParameter('yrange', []);
p.addParameter('xrange', []);
p.addParameter('smoothSigma', 1.15);
p.addParameter('maxShiftFrac', 0.1);
p.addParameter('maxShiftNR', 5);
p.addParameter('plot', true);
p.addParameter('rawMov', []);
p.addParameter('rawPC', false);
p.addParameter('kymoRow', []);
p.addParameter('kymoCol', []);
p.addParameter('kymoWidth', 5);
p.addParameter('kymoBin', 5);
p.addParameter('hpSigma', 10);
p.addParameter('maxFrames', Inf);
p.addParameter('chunk', 200);
p.parse(varargin{:});
o = p.Results;
rawMov = o.rawMov;
o.rawMov = [];      % keep the movie out of M.opts

src = open_source(mov, o.maxFrames, 'movie');
[Ly, Lx, T] = deal(src.Ly, src.Lx, src.T);
raw = [];
if ~isempty(rawMov)
    raw = open_source(rawMov, o.maxFrames, 'raw movie');
    if ~isequal([raw.Ly raw.Lx], [Ly Lx])
        error(['rawMov is %dx%d px but mov is %dx%d px; give the pixels that match ' ...
            '(e.g. struct(''path'', rawTif, ''rows'', S.outRows, ''cols'', S.outCols) ' ...
            'with S from the _shifts.mat).'], raw.Ly, raw.Lx, Ly, Lx);
    end
end

% ---- crop to valid region
if isempty(o.yrange), o.yrange = [1 Ly]; end
if isempty(o.xrange), o.xrange = [1 Lx]; end
yr = o.yrange(1):o.yrange(2);
xr = o.xrange(1):o.xrange(2);
[ly, lx] = deal(numel(yr), numel(xr));

% ---- 1-5. PC metric on the sampled frames (sample mean = mean image)
frameIdx = sample_index(T, Ly, Lx);
fprintf('  reading %d sample frames ...\n', numel(frameIdx));
X = read_frames(src, frameIdx, yr, xr);
meanImg = mean_image(X);
[regDX, tPC, topMean, botMean] = pc_metric(X, o);
clear X

[frameCorrRaw, meanImgRaw, frameIdxRaw] = deal([]);
[regDXRaw, tPCRaw, topMeanRaw, botMeanRaw] = deal([]);
if ~isempty(raw)
    frameIdxRaw = sample_index(raw.T, Ly, Lx);
    fprintf('  reading %d raw sample frames ...\n', numel(frameIdxRaw));
    Xr = read_frames(raw, frameIdxRaw, yr, xr);
    meanImgRaw = mean_image(Xr);
    if o.rawPC
        [regDXRaw, tPCRaw, topMeanRaw, botMeanRaw] = pc_metric(Xr, o);
    end
    clear Xr
end

% ---- 7. kymograph lines (picked on the mean image)
h = floor(o.kymoWidth / 2);
[kRow, kCol] = deal(o.kymoRow, o.kymoCol);
if isempty(kRow) || isempty(kCol)
    [aRow, aCol] = pick_kymo_lines(meanImg, h, o.hpSigma);
    if isempty(kRow), kRow = aRow; end
    if isempty(kCol), kCol = aCol; end
end
kRow = min(max(round(kRow), 1 + h), ly - h);
kCol = min(max(round(kCol), 1 + h), lx - h);

% ---- 6-7. streaming pass: frame-to-mean correlation and line profiles
fprintf('  streaming %d frames ...\n', T);
[frameCorr, H, V] = stream_pass(src, yr, xr, meanImg, kRow, kCol, h, o.chunk);
kymo = struct('row', kRow, 'col', kCol, 'width', 2*h + 1, 'bin', o.kymoBin);
[kymo.H, kymo.V, kymo.t] = bin_profiles(H, V, o.kymoBin);
[kymo.Hraw, kymo.Vraw, kymo.tRaw] = deal([]);
if ~isempty(raw)
    fprintf('  streaming %d raw frames ...\n', raw.T);
    [frameCorrRaw, H, V] = stream_pass(raw, yr, xr, meanImgRaw, kRow, kCol, h, o.chunk);
    [kymo.Hraw, kymo.Vraw, kymo.tRaw] = bin_profiles(H, V, o.kymoBin);
end
clear H V

rawSize = [];
if ~isempty(raw), rawSize = [raw.Ly raw.Lx raw.T]; end
M = struct('regDX', regDX, 'tPC', tPC, 'topMean', topMean, ...
           'botMean', botMean, 'frameIdx', frameIdx, 'meanImg', meanImg, ...
           'frameCorr', frameCorr, 'movSize', [Ly Lx T], ...
           'frameCorrRaw', frameCorrRaw, 'meanImgRaw', meanImgRaw, 'rawSize', rawSize, ...
           'regDXRaw', regDXRaw, 'tPCRaw', tPCRaw, 'topMeanRaw', topMeanRaw, ...
           'botMeanRaw', botMeanRaw, 'frameIdxRaw', frameIdxRaw, ...
           'kymo', kymo, 'opts', o);

if o.plot
    plot_registration_metrics(M);
    plot_frame_corr(M);
    plot_kymographs(M);
end
end

% =====================================================================
function src = open_source(m, maxFrames, label)
% Movie source: an array, a tif path, or struct('path', ..., 'rows', ...,
% 'cols', ...). Fields: kind ('array' / 'tif'), data or path, rows, cols
% (pixels of the stored frame that are used), Ly, Lx, T, cls.
src = struct('kind', 'array', 'data', [], 'path', '', 'rows', [], 'cols', [], ...
             'Ly', 0, 'Lx', 0, 'T', 0, 'cls', '');
if isnumeric(m) || islogical(m)
    src.data = m;
    [H, W, n] = size(m);
    src.cls = class(m);
else
    if ischar(m) || isstring(m)
        m = struct('path', char(m));
    end
    if ~isstruct(m) || ~isfield(m, 'path') || ~isfile(m.path)
        error('%s: give an array, an existing tif path, or a struct with a ''path'' field.', label);
    end
    src.kind = 'tif';
    src.path = char(m.path);
    if isfield(m, 'rows'), src.rows = m.rows(:)'; end
    if isfield(m, 'cols'), src.cols = m.cols(:)'; end
    [H, W, n, src.cls] = tif_info(src.path, maxFrames);
end
if isempty(src.rows), src.rows = 1:H; end
if isempty(src.cols), src.cols = 1:W; end
if max(src.rows) > H || max(src.cols) > W
    error('%s: rows/cols exceed the %dx%d frame.', label, H, W);
end
src.Ly = numel(src.rows);
src.Lx = numel(src.cols);
src.T = min(n, maxFrames);
end

function [H, W, n, cls] = tif_info(path, maxFrames)
% Frame size, class and number of pages (up to maxFrames) of a tif,
% walking the page headers without reading image data.
[t, cleanup] = open_tif(path); %#ok<ASGLU>
H = t.getTag('ImageLength');
W = t.getTag('ImageWidth');
cls = class(t.read());
n = 1;
while n < maxFrames && ~t.lastDirectory()
    t.nextDirectory();
    n = n + 1;
end
end

function [t, cleanup] = open_tif(path)
% Tiff object with libtiff warnings off; both restored by cleanup.
w = warning('off', 'imageio:tiffmexutils:libtiffWarning');
t = Tiff(path, 'r');
cleanup = onCleanup(@() close_tif(t, w));
end

function close_tif(t, w)
t.close();
warning(w);
end

function idx = sample_index(T, Ly, Lx)
% Frames for the PC metric: 2000 if large frame or short movie, else 5000.
if max(Ly, Lx) > 700 || T < 5000
    nSamp = min(2000, T);
else
    nSamp = 5000;
end
idx = unique(round(linspace(1, T, nSamp)));
end

function X = read_frames(src, idx, yr, xr)
% Frames idx (ascending) cropped to yr, xr of the source pixels, in the
% source's class. A tif is walked page by page; only frames in idx are read.
R = src.rows(yr);
C = src.cols(xr);
if strcmp(src.kind, 'array')
    X = src.data(R, C, idx);
    return
end
X = zeros(numel(R), numel(C), numel(idx), src.cls);
[t, cleanup] = open_tif(src.path); %#ok<ASGLU>
cur = 1;
for k = 1:numel(idx)
    while cur < idx(k)
        t.nextDirectory();
        cur = cur + 1;
    end
    I = t.read();
    X(:,:,k) = I(R, C);
end
end

function mimg = mean_image(X)
% Mean over frames in double, a frame block at a time (no full double copy).
mimg = zeros(size(X, 1), size(X, 2));
for s = 1:500:size(X, 3)
    mimg = mimg + sum(X(:,:,s:min(s+499, end)), 3, 'double');
end
mimg = mimg / size(X, 3);
end

function [regDX, tPC, topMean, botMean] = pc_metric(X, o)
% Steps 2-5 on sampled frames X (ly x lx x nf). PCA via the nf x nf Gram
% matrix of the pixel-mean-subtracted frames, accumulated over pixel blocks
% (same PC time courses as an SVD of the nf x npix matrix).
[ly, lx, nf] = size(X);
nPC  = min(o.nPC, nf - 1);
nTop = min(o.nTop, floor(nf/2));
Xp = reshape(X, ly*lx, nf);
npix = ly*lx;
step = max(1, floor(2e8 / (8 * nf)));          % ~200 MB of doubles per block
G = zeros(nf);
for p0 = 1:step:npix
    pix = p0:min(p0 + step - 1, npix);
    B = double(Xp(pix, :));
    B = B - mean(B, 2);
    G = G + B' * B;
end
clear Xp B
G = (G + G') / 2;
[U, L] = eig(G, 'vector');
[L, order] = sort(L, 'descend');
U = U(:, order(1:nPC));
tPC = U .* sqrt(max(L(1:nPC), 0))';             % nf x nPC, = U*S of the SVD

% ---- 3-4. top/bottom means and their relative offsets
topMean = zeros(ly, lx, nPC, 'single');
botMean = zeros(ly, lx, nPC, 'single');
regDX   = zeros(nPC, 4);
maxShift = round(o.maxShiftFrac * min(ly, lx));
for k = 1:nPC
    [~, ord] = sort(tPC(:, k));
    botMean(:,:,k) = mean(X(:,:,ord(1:nTop)), 3);
    topMean(:,:,k) = mean(X(:,:,ord(end-nTop+1:end)), 3);

    ref = topMean(:,:,k);  img = botMean(:,:,k);
    % rigid
    [dy, dx] = phasecorr_shift(ref, img, maxShift, o.smoothSigma);
    rig = [dy dx];
    img = fourier_shift(img, dy, dx);         % remove rigid part
    % non-rigid on overlapping blocks
    nr = nonrigid_shifts(ref, img, o.blockSize, o.maxShiftNR, o.smoothSigma);
    nrLen  = sqrt(sum(nr.^2, 2));
    totLen = sqrt(sum((nr + rig).^2, 2));
    regDX(k,:) = [norm(rig), mean(nrLen), max(nrLen), mean(totLen)];
end
end

function [r, H, V] = stream_pass(src, yr, xr, mimg, row, col, h, chunk)
% One pass over all frames, chunk at a time: Pearson correlation of each
% (cropped) frame with mimg, and the band-averaged profile along the
% horizontal (H, lx x T) and vertical (V, ly x T) kymograph lines.
T = src.T;
R = src.rows(yr);
C = src.cols(xr);
[ly, lx] = deal(numel(R), numel(C));
mz = mimg(:) - mean(mimg(:));
mz = mz / norm(mz);
r = zeros(T, 1);
H = zeros(lx, T, 'single');
V = zeros(ly, T, 'single');
isTif = strcmp(src.kind, 'tif');
if isTif
    [t, cleanup] = open_tif(src.path); %#ok<ASGLU>
end
nextReport = 0.1;
for s = 1:chunk:T
    f = s:min(s + chunk - 1, T);
    if isTif
        F = zeros(ly, lx, numel(f), src.cls);
        for k = 1:numel(f)
            if f(k) > 1
                t.nextDirectory();
            end
            I = t.read();
            F(:,:,k) = I(R, C);
        end
    else
        F = src.data(R, C, f);
    end
    H(:, f) = reshape(mean(single(F(row-h:row+h, :, :)), 1), lx, []);
    V(:, f) = reshape(mean(single(F(:, col-h:col+h, :)), 2), ly, []);
    D = double(reshape(F, [], numel(f)));
    D = D - mean(D, 1);
    r(f) = (mz' * D) ./ sqrt(sum(D.^2, 1));
    if f(end) / T >= nextReport
        fprintf('    %3.0f%%\n', 100 * f(end) / T);
        nextReport = nextReport + 0.1;
    end
end
end

function [H, V, t] = bin_profiles(H, V, nBin)
% Average line profiles (position x frame) over bins of nBin frames.
% t = centre frame of each bin.
T = size(H, 2);
nb = floor(T / nBin);
if nb < 1
    nBin = T;
    nb = 1;
end
H = reshape(mean(reshape(H(:, 1:nb*nBin), size(H, 1), nBin, nb), 2), size(H, 1), nb);
V = reshape(mean(reshape(V(:, 1:nb*nBin), size(V, 1), nBin, nb), 2), size(V, 1), nb);
t = ((1:nb) - 0.5) * nBin + 0.5;
end

function [row, col] = pick_kymo_lines(mimg, h, sig)
% Row and column whose band (+/- h px) crosses the most structure: highest
% variance of the high-pass filtered mean image. Avoids the outer edges,
% where the high-pass filter is unreliable.
k = exp(-(-ceil(3*sig):ceil(3*sig)).^2 / (2*sig^2));
k = k / sum(k);
blur = @(x) conv2(k, k, x, 'same');
hp = mimg - blur(mimg) ./ blur(ones(size(mimg)));   % normalized at the edges
[ly, lx] = size(mimg);
m = min([h + ceil(sig), floor(ly/4), floor(lx/4)]);
band = ones(2*h + 1, 1) / (2*h + 1);
rowScore = conv(var(hp(:, m+1:lx-m), 0, 2), band, 'same');
colScore = conv(var(hp(m+1:ly-m, :), 0, 1)', band, 'same');
rowScore([1:m, ly-m+1:ly]) = -Inf;
colScore([1:m, lx-m+1:lx]) = -Inf;
[~, row] = max(rowScore);
[~, col] = max(colScore);
end

% =====================================================================
function [dy, dx, cmax] = phasecorr_shift(ref, img, maxShift, sig)
% Phase correlation: whiten both spectra, multiply, smooth in Fourier
% domain with a Gaussian, find peak within +/- maxShift, subpixel refine.
[ly, lx] = size(ref);
tap = spatial_taper(ly, lx, 2*sig);
ref = clip_pct(ref);  img = clip_pct(img);
Fr = fft2((ref - mean(ref(:))) .* tap);
Fi = fft2((img - mean(img(:))) .* tap);
Fr = Fr ./ (abs(Fr) + 1e-5);
Fi = Fi ./ (abs(Fi) + 1e-5);
G  = gauss_fourier(ly, lx, sig);
cc = real(ifft2(Fr .* conj(Fi) .* G));
cc = fftshift(cc);
cy = floor(ly/2) + 1;  cx = floor(lx/2) + 1;
maxShift = max(1, min([maxShift, cy-2, cx-2]));
win = cc(cy-maxShift:cy+maxShift, cx-maxShift:cx+maxShift);
[cmax, idx] = max(win(:));
[iy, ix] = ind2sub(size(win), idx);
% 3-point parabolic subpixel refinement (Suite2p uses Kriging; same idea)
sy = subpix(win, iy, ix, 1);  sx = subpix(win, iy, ix, 2);
dy = iy - maxShift - 1 + sy;
dx = ix - maxShift - 1 + sx;
end

function s = subpix(w, iy, ix, dim)
s = 0;
if dim == 1
    if iy > 1 && iy < size(w,1), a = w(iy-1,ix); b = w(iy,ix); c = w(iy+1,ix); else, return; end
else
    if ix > 1 && ix < size(w,2), a = w(iy,ix-1); b = w(iy,ix); c = w(iy,ix+1); else, return; end
end
den = a - 2*b + c;
if den ~= 0, s = 0.5 * (a - c) / den; end
s = max(-0.5, min(0.5, s));
end

function nr = nonrigid_shifts(ref, img, bs, maxNR, sig)
% Overlapping (>=50%) square blocks; phase-correlate each block.
[ly, lx] = size(ref);
bs = min(bs, [ly lx]);
ny = max(1, ceil(1.5 * ly / bs(1)));  nx = max(1, ceil(1.5 * lx / bs(2)));
ys = round(linspace(1, ly - bs(1) + 1, ny));
xs = round(linspace(1, lx - bs(2) + 1, nx));
nr = zeros(numel(ys)*numel(xs), 2);  n = 0;
for iy = ys
    for ix = xs
        n = n + 1;
        r = ref(iy:iy+bs(1)-1, ix:ix+bs(2)-1);
        m = img(iy:iy+bs(1)-1, ix:ix+bs(2)-1);
        [nr(n,1), nr(n,2)] = phasecorr_shift(r, m, maxNR, sig);
    end
end
end

function out = fourier_shift(img, dy, dx)
[ly, lx] = size(img);
ky = ifftshift((0:ly-1) - floor(ly/2))' / ly;
kx = ifftshift((0:lx-1) - floor(lx/2)) / lx;
out = real(ifft2(fft2(img) .* exp(-2i*pi*(ky*dy + kx*dx))));
end

function G = gauss_fourier(ly, lx, sig)
ky = ifftshift((0:ly-1) - floor(ly/2))' / ly;
kx = ifftshift((0:lx-1) - floor(lx/2)) / lx;
G = exp(-2*pi^2*sig^2*(ky.^2 + kx.^2));
end

function tap = spatial_taper(ly, lx, slope)
my = 1 ./ (1 + exp(-(((ly-1)/2 - abs((0:ly-1)' - (ly-1)/2)) - 2*slope) / slope));
mx = 1 ./ (1 + exp(-(((lx-1)/2 - abs((0:lx-1)  - (lx-1)/2)) - 2*slope) / slope));
tap = my * mx;
end

function x = clip_pct(x)
v = sort(x(:));
lo = v(max(1, round(0.01*numel(v))));  hi = v(max(1, round(0.99*numel(v))));
x = min(max(x, lo), hi);
end

% =====================================================================
function plot_registration_metrics(M)
d = M.topMean(:,:,1) - M.botMean(:,:,1);
f1 = figure('Name', 'PC1 top/bottom', 'Color', 'w', 'Position', [100 100 1200 380]);
subplot(1,3,1); imagesc(M.topMean(:,:,1)); axis image off; colormap(gca, gray); title('PC1 top-frames mean');
subplot(1,3,2); imagesc(M.botMean(:,:,1)); axis image off; colormap(gca, gray); title('PC1 bottom-frames mean');
subplot(1,3,3); imagesc(d, max(abs(d(:)))*[-1 1]); axis image off; colormap(gca, bwr_map); colorbar; title('top - bottom');

f2 = figure('Name', 'PC1 time course', 'Color', 'w', 'Position', [100 520 900 250]);
plot(M.frameIdx, M.tPC(:,1), 'k'); xlabel('frame'); ylabel('PC1 projection'); box off; title('PC1 across frames');

f3 = figure('Name', 'Registration offsets', 'Color', 'w', 'Position', [1000 520 600 400]);
plot(M.regDX, '.-', 'MarkerSize', 12); box off;
legend({'rigid', 'mean NR', 'max NR', 'mean(rigid+NR)'}, 'Location', 'northeast'); legend boxoff;
xlabel('PC #'); ylabel('offset length (px)'); title(sprintf('max over PCs: %.3f px', max(M.regDX(:,4))));
drawnow;
M.figs = [f1 f2 f3];
end

function plot_frame_corr(M)
% Frame-to-mean correlation over time and its distribution (raw vs
% corrected if the raw movie was given).
hasRaw = ~isempty(M.frameCorrRaw);
figure('Name', 'Frame-to-mean correlation', 'Color', 'w', 'Position', [100 100 1200 330]);
subplot(1,4,1:3); hold on
if hasRaw
    plot(M.frameCorrRaw, 'Color', [0.85 0.33 0.10]);
end
plot(M.frameCorr, 'k');
xlabel('frame'); ylabel('corr. with mean image'); box off
ok = ~isnan(M.frameCorr);    % NaN = flat frame
title(sprintf('median %.3f, 1st pct %.3f (corrected)', median(M.frameCorr(ok)), prctile1(M.frameCorr(ok))));
subplot(1,4,4); hold on
edges = linspace(min([M.frameCorr; M.frameCorrRaw; 0.99]), 1, 60);
if hasRaw
    histogram(M.frameCorrRaw, edges, 'FaceColor', [0.85 0.33 0.10], 'EdgeColor', 'none');
end
histogram(M.frameCorr, edges, 'FaceColor', 'k', 'EdgeColor', 'none');
xlabel('corr. with mean image'); ylabel('frames'); box off
if hasRaw
    legend({'raw', 'corrected'}, 'Location', 'northwest'); legend boxoff
end
drawnow;
end

function plot_kymographs(M)
% Mean image with the two lines, and the kymograph along each (raw above
% corrected if the raw movie was given). Time runs left to right.
K = M.kymo;
hasRaw = ~isempty(K.Hraw);
nr = 2 + 2*hasRaw;
figure('Name', 'Kymographs', 'Color', 'w', 'Position', [100 100 1400 220*nr]);
subplot(nr, 4, 1 + 4*(0:nr-1));
imagesc(M.meanImg); axis image off; colormap(gca, gray); hold on
h = (K.width - 1) / 2;
[ly, lx] = size(M.meanImg);
patch([0.5 lx+0.5 lx+0.5 0.5], K.row + [-h -h h h] + [-0.5 -0.5 0.5 0.5], 'c', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'c');
patch(K.col + [-h h h -h] + [-0.5 0.5 0.5 -0.5], [0.5 0.5 ly+0.5 ly+0.5], 'm', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'm');
title({'mean image', 'cyan = horizontal line, magenta = vertical'});
panels = {K.H, K.t, 'horizontal line (cyan)', 'x (px)'; ...
          K.V, K.t, 'vertical line (magenta)', 'y (px)'};
if hasRaw
    panels = [{K.Hraw, K.tRaw, 'RAW horizontal line (cyan)', 'x (px)'}; panels(1,:); ...
              {K.Vraw, K.tRaw, 'RAW vertical line (magenta)', 'y (px)'}; panels(2,:)];
    panels(2:2:4, 3) = strcat({'CORRECTED '}, panels(2:2:4, 3));
end
for i = 1:nr
    [img, t, ttl, yl] = panels{i, :};
    subplot(nr, 4, (2:4) + 4*(i-1));
    imagesc(t([1 end]), [1 size(img, 1)], img, prctile1(img(:), [1 99]));
    colormap(gca, gray); box off
    ylabel(yl); title(sprintf('%s  (%d px band, %d-frame bins)', ttl, K.width, K.bin));
    if i == nr, xlabel('frame'); end
end
drawnow;
end

function v = prctile1(x, q)
% Percentile(s) without the Statistics Toolbox (default 1st).
if nargin < 2, q = 1; end
x = sort(double(x(:)));
if isscalar(x)
    v = x * ones(size(q));
else
    v = interp1(linspace(0, 100, numel(x)), x, q);
end
if numel(v) == 2 && v(1) == v(2), v = v(1) + [-1 1]; end   % flat image: valid clim
end

function c = bwr_map
n = 128; a = linspace(0,1,n)';
c = [[a a ones(n,1)]; [ones(n,1) flipud(a) flipud(a)]];
end
