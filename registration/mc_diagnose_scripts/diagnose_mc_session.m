% diagnose_mc_session.m
%
% Read-only diagnostics for one motion-corrected session: is a slow change
% (e.g. PC1 drifting over the recording) photobleaching, xy motion, axial
% drift, or a real baseline change?
%
% Inputs: the ConcatenatedMovie_MC.tif of a session (its _shifts.mat is read
% automatically). The raw .sbx files are taken from the _shifts.mat 'files'
% field unless you list them in sbxFiles.
%
% What it reports / plots (all subsampled by 'stride' frames):
%   Fig 1  time courses (segment boundaries = dashed lines)
%            a. mean fluorescence per frame: raw .sbx vs motion-corrected
%               (same output pixels, normalised to the first time bin).
%               Decay = bleaching; rise/steps = signal, z, laser, segments.
%            b. rigid shifts dy, dx applied by motion correction
%            c. phase-correlation peak (cmax) and outlier frames
%            d. correlation of every frame with the session mean image
%               (falls where structure changes: z drift, motion, blur)
%   Fig 2  early vs late mean image, their difference, and:
%            - sharpness of each time bin's mean image
%            - correlation matrix between time-bin mean images
%            - xy shift between first and last bin
%            - "gain vs shift" check: correlation of (last-first) with the
%              mean image (intensity change) and with its spatial gradient
%              (a shift leaves a dipole along the gradient)
%   Text   bidirectional scan phase offset estimated from raw frames
%          (even vs odd rows); |lag| > ~0.5 px blurs the mean image and
%          motion correction cannot fix it.
%
% Needs: sbx_method.m (registration folder), Image Processing Toolbox.

%%%%%%%%%%%%%%%%%%
%%% USER INPUT %%%
%%%%%%%%%%%%%%%%%%
clear; close all
mcTif      = "Y:\2p_animals\mouse1\261008\932_1dir_middle_rec001\ConcatenatedMovie_MC.tif";
sbxFiles   = {"Y:\2p_animals\mouse1\261008\ac_001_001.sbx", 'Y:\2p_animals\mouse1\261008\ac_001_002.sbx'};      % {} = use the 'files' saved in the _shifts.mat; or {'Y:\...\261009_001_000', ...}
sbxChannel = 1;       % PMT channel of the functional movie
stride     = 5;       % read every Nth frame (1 = all; slower)
nBins      = 6;       % time bins for the mean-image comparison
runRaw     = true;    % raw-.sbx comparison + bidirectional phase check
nBidiFrames = 200;    % frames used for the bidirectional phase estimate
smoothWin  = 50;      % moving-mean window (in sampled frames) for the plots
outDir     = '';      % '' = <mcTif folder>\diagnostics
%%%%%%%%%%%%%%%%%%%%%%%
%%% END: USER INPUT %%%
%%%%%%%%%%%%%%%%%%%%%%%

here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
addpath(genpath(fileparts(here)));     % parent folder: sbx_method, mc_diag_method

[d, n] = fileparts(mcTif);
sf = fullfile(d, [n '_shifts.mat']);
if isempty(outDir), outDir = fullfile(d, 'diagnostics'); end
if ~isfolder(outDir), mkdir(outDir); end

%% shifts
S = struct('dy', [], 'dx', [], 'cmax', [], 'bad', [], 'frameIndex', [], ...
    'files', [], 'crop', [], 'meanImg', [], 'bidi', []);
if isfile(sf)
    vars = who('-file', sf);
    S = mergeStruct(S, load(sf, vars{ismember(vars, fieldnames(S))}));
end

%% motion-corrected movie: strided pass
t = Tiff(mcTif, 'r');
closeT = onCleanup(@() t.close());
H = t.getTag('ImageLength');
W = t.getTag('ImageWidth');
N = numel(S.dy);
if N == 0 && ~isempty(S.frameIndex), N = size(S.frameIndex, 1); end
if N == 0, N = numel(imfinfo(mcTif)); end
idx = 1:stride:N;
binOf = min(nBins, ceil(idx / N * nBins));
meanMC = nan(numel(idx), 1);
binSum = zeros(H, W, nBins);
binCnt = zeros(nBins, 1);
Mref = [];
if ~isempty(S.meanImg) && isequal(size(S.meanImg), [H W])
    Mref = double(S.meanImg(:));
    Mref = Mref - mean(Mref);
end
frameCorr = nan(numel(idx), 1);
fprintf('Reading %d of %d frames from %s ...\n', numel(idx), N, n);
t.setDirectory(1);
cur = 1;
for k = 1:numel(idx)
    while cur < idx(k)
        t.nextDirectory();
        cur = cur + 1;
    end
    f = double(t.read());
    meanMC(k) = mean(f(:));
    binSum(:, :, binOf(k)) = binSum(:, :, binOf(k)) + f;
    binCnt(binOf(k)) = binCnt(binOf(k)) + 1;
    if ~isempty(Mref)
        v = f(:) - meanMC(k);
        frameCorr(k) = (v' * Mref) / (norm(v) * norm(Mref) + eps);
    end
    if mod(k, 1000) == 0, fprintf('  %d / %d\n', k, numel(idx)); end
end
clear closeT
binImg = binSum ./ reshape(max(binCnt, 1), 1, 1, []);

% segment boundaries (frame number where a new .sbx file starts)
segBreaks = [];
if ~isempty(S.frameIndex)
    segBreaks = find(diff(S.frameIndex(:, 1)) ~= 0) + 1;
end

%% raw .sbx: per-frame mean in the output pixels, bidirectional phase
meanRaw = nan(numel(idx), 1);
bidi = struct('lag', nan, 'cc', [], 'lags', []);
if runRaw
    files = sbxFiles;
    if isempty(files) && ~isempty(S.files), files = cellstr(S.files); end
    files = cellfun(@(f) regexprep(char(f), '\.sbx$', '', 'ignorecase'), files, 'UniformOutput', false);
    okFiles = ~isempty(files) && all(cellfun(@(f) isfile([f '.sbx']), files));
    if ~okFiles && ~isempty(files)
        fprintf('  ! raw .sbx not found, e.g. %s.sbx (set sbxFiles)\n', files{1});
    end
    if okFiles && ~isempty(S.frameIndex) && ~isempty(S.crop)
        info = cellfun(@sbx_method.info, files, 'UniformOutput', false);
        info = [info{:}];
        ci = find(info(1).chs == sbxChannel);
        o = double(S.crop.out);                 % [r1 c1 r2 c2] output rect, raw px
        fi = S.frameIndex(idx, :);
        for fIdx = 1:numel(info)
            sel = find(fi(:, 1) == fIdx);
            if isempty(sel), continue, end
            fid = fopen(info(fIdx).path, 'r');
            closeF = onCleanup(@() fclose(fid));
            for k = sel(:)'
                x = readRaw(fid, info(fIdx), fi(k, 2), ci);
                r = x(o(1):o(3), o(2):o(4));
                meanRaw(k) = mean(double(r(:)));
            end
            clear closeF
        end
        % bidirectional phase from evenly spaced frames of the first file
        rcRaw = double(S.crop.raw(:)');
        rc = [max(rcRaw(1:2), 1), min(rcRaw(3:4), [info(1).rows info(1).cols])];
        nF = info(1).nFramesDisk;
        pick = unique(round(linspace(2, nF, min(nBidiFrames, nF - 1))));
        fid = fopen(info(1).path, 'r');
        closeF = onCleanup(@() fclose(fid));
        M = 0;
        for k = pick
            x = readRaw(fid, info(1), k - 1, ci);
            M = M + double(x(rc(1):rc(3), rc(2):rc(4)));
        end
        clear closeF
        bidi = bidiPhase(M / numel(pick), rc(1));
    else
        fprintf('  ! raw .sbx files not found (or no frameIndex/crop in shifts); raw section skipped\n');
    end
end

%% summaries
sharp = arrayfun(@(b) sharpness(binImg(:, :, b)), 1:nBins);
c = nan(nBins);
ctr = {round(H*0.1):round(H*0.9), round(W*0.1):round(W*0.9)};
for a = 1:nBins
    for b = 1:nBins
        x = binImg(ctr{1}, ctr{2}, a); y = binImg(ctr{1}, ctr{2}, b);
        c(a, b) = corr(x(:), y(:));
    end
end
first = binImg(:, :, 1);
last = binImg(:, :, nBins);
[sy, sx] = phaseShift(first, last);
diffImg = last - first;
[gx, gy] = imgradientxy(0.5 * (first + last), 'central');
mid = {ctr{1}, ctr{2}};
cGain = corr(vec(diffImg(mid{:})), vec(0.5 * (first(mid{:}) + last(mid{:}))));
cShift = max(abs([corr(vec(diffImg(mid{:})), vec(gx(mid{:}))), ...
                  corr(vec(diffImg(mid{:})), vec(gy(mid{:})))]));

fprintf('\n================ %s ================\n', n);
fprintf('frames: %d  (sampled %d, stride %d)\n', N, numel(idx), stride);
if ~isempty(S.dy)
    fprintf('rigid shifts: dy %.2f +- %.2f px, dx %.2f +- %.2f px, range dy [%.1f %.1f] dx [%.1f %.1f]\n', ...
        mean(S.dy), std(S.dy), mean(S.dx), std(S.dx), min(S.dy), max(S.dy), min(S.dx), max(S.dx));
    fprintf('  slow component: dy changes %.2f px, dx %.2f px between first and last bin\n', ...
        mean(S.dy(end-round(N/nBins)+1:end)) - mean(S.dy(1:round(N/nBins))), ...
        mean(S.dx(end-round(N/nBins)+1:end)) - mean(S.dx(1:round(N/nBins))));
end
if ~isempty(S.bad), fprintf('outlier frames: %.2f %%\n', 100 * mean(S.bad)); end
b1 = mean(meanMC(binOf == 1));
bN = mean(meanMC(binOf == nBins));
fprintf('mean F (MC movie): last bin / first bin = %.3f\n', bN / b1);
if any(~isnan(meanRaw))
    r1 = mean(meanRaw(binOf == 1)); rN = mean(meanRaw(binOf == nBins));
    fprintf('mean F (raw .sbx): last bin / first bin = %.3f\n', rN / r1);
end
fprintf('first->last bin mean image shift: dy %.2f px, dx %.2f px\n', sy, sx);
fprintf('  corr(last-first, mean image)   = %+.2f   (intensity / gain change)\n', cGain);
fprintf('  max |corr(last-first, gradient)| = %.2f   (shift dipole)\n', cShift);
fprintf('sharpness per time bin: %s\n', sprintf('%.3f ', sharp));
if ~isnan(bidi.lag)
    fprintf('bidirectional row offset, RAW .sbx (odd vs even rows): %+.2f px%s\n', bidi.lag, ...
        ternary(abs(bidi.lag) > 0.5, '   <-- CHECK: likely blurring the mean image', '   (ok)'));
end
if isstruct(S.bidi)
    fprintf(['bidi recorded by motion correction: mode %s, estimated %+.2f px, applied [odd %+.2f, even %+.2f] px, ' ...
        'residual %+.2f px\n'], S.bidi.mode, S.bidi.lag, S.bidi.applied(1), S.bidi.applied(2), S.bidi.lagAfter);
else
    fprintf('bidi: no record in the _shifts.mat (motion correction run without bidiShift)\n');
end
if ~isempty(S.crop)
    resid = bidiPhase(mean(binImg, 3), double(S.crop.out(1)));
    fprintf('bidirectional row offset left in the MC movie mean image: %+.2f px%s\n', resid.lag, ...
        ternary(abs(resid.lag) > 0.5, '   <-- CHECK: not corrected', '   (ok)'));
end
fprintf(['Reading it: F falling over time = bleaching; F rising or stepping = signal / z / laser;\n' ...
    '  large gradient-corr or shift = xy motion; high gain-corr + falling frame-corr = axial drift.\n']);

%% Fig 1: time courses
x = idx;
f1 = figure('Name', 'time courses', 'Position', [100 100 1100 800]);
tl = tiledlayout(4, 1, 'TileSpacing', 'compact');
nexttile; hold on
plot(x, meanMC / b1, 'Color', [.75 .75 .75]);
plot(x, movmean(meanMC / b1, smoothWin), 'k', 'LineWidth', 1.2);
if any(~isnan(meanRaw))
    plot(x, movmean(meanRaw / r1, smoothWin), 'r', 'LineWidth', 1.2);
    legend({'MC (raw samples)', 'MC (smoothed)', 'raw .sbx (smoothed)'}, 'Location', 'best');
end
ylabel('mean F / first bin'); title('mean fluorescence per frame'); segLines(segBreaks); box off
nexttile; hold on
if ~isempty(S.dy)
    hy = plot(1:N, S.dy, 'b'); hx = plot(1:N, S.dx, 'r'); legend([hy hx], {'dy', 'dx'}, 'Location', 'best');
end
ylabel('rigid shift (px)'); segLines(segBreaks); box off
nexttile; hold on
if ~isempty(S.cmax)
    plot(1:N, S.cmax, 'k');
    if ~isempty(S.bad) && any(S.bad), plot(find(S.bad), S.cmax(S.bad), 'r.'); end
end
ylabel('corr peak (cmax)'); segLines(segBreaks); box off
nexttile; hold on
plot(x, frameCorr, 'Color', [.75 .75 .75]); plot(x, movmean(frameCorr, smoothWin), 'k', 'LineWidth', 1.2);
ylabel('corr with mean image'); xlabel('frame'); segLines(segBreaks); box off
linkaxes(findall(f1, 'Type', 'axes'), 'x'); xlim([1 N]);
title(tl, strrep(n, '_', '\_'));
saveas(f1, fullfile(outDir, 'diag_timecourses.png'));

%% Fig 2: early vs late
f2 = figure('Name', 'early vs late', 'Position', [100 100 1300 750]);
tl2 = tiledlayout(2, 3, 'TileSpacing', 'compact');
lim = prctile([first(:); last(:)], [1 99.5]);
nexttile; imagesc(first, lim); axis image off; colormap(gca, gray); title(sprintf('first bin (frames 1-%d)', round(N/nBins)));
nexttile; imagesc(last, lim); axis image off; colormap(gca, gray); title(sprintf('last bin (frames %d-%d)', N - round(N/nBins) + 1, N));
nexttile; m = prctile(abs(diffImg(:)), 99.5); imagesc(diffImg, [-m m]); axis image off; colormap(gca, bwr()); colorbar; title('last - first');
nexttile; plot(1:nBins, sharp, 'o-'); xlabel('time bin'); ylabel('sharpness (norm. gradient)'); title('mean-image sharpness'); box off
nexttile; imagesc(c, [min(c(:)) 1]); axis image; colorbar; colormap(gca, parula); title('bin-to-bin image correlation'); xlabel('bin'); ylabel('bin');
nexttile; plot(x, movmean(frameCorr, smoothWin), 'k'); hold on; segLines(segBreaks); xlabel('frame'); ylabel('corr with mean'); title('frame / mean correlation'); box off
title(tl2, strrep(n, '_', '\_'));
saveas(f2, fullfile(outDir, 'diag_early_vs_late.png'));

save(fullfile(outDir, 'diag_results.mat'), 'idx', 'meanMC', 'meanRaw', 'frameCorr', 'binImg', ...
    'sharp', 'c', 'sy', 'sx', 'cGain', 'cShift', 'bidi', 'segBreaks');
fprintf('Saved figures + diag_results.mat in %s\n', outDir);


%% Local functions
function x = readRaw(fid, si, f0, ci)
    % one frame (0-based f0) of one channel as [rows cols] uint16, sbxread-style
    fseek(fid, f0 * si.frameBytes, 'bof');
    x = fread(fid, si.frameBytes / 2, 'uint16=>uint16');
    x = reshape(x, si.nchan, si.cols, si.rows);
    x = intmax('uint16') - permute(x(ci, :, :), [3 2 1]);
end

function b = bidiPhase(M, r0)
    % column lag between odd and even raw rows of a (mean) image whose first
    % row is raw-frame row r0 (>0: odd-row features lie right of even-row ones;
    % same convention as sbx_motion_correct out.bidi.lag)
    iA = find(mod(r0 - 1 + (1:size(M, 1)), 2) == 1);
    iA = iA(iA < size(M, 1));
    A = M(iA, :);  B = M(iA + 1, :);
    A = A - mean(A, 2);   B = B - mean(B, 2);
    lags = -8:8;
    cc = zeros(size(lags));
    for i = 1:numel(lags)
        l = lags(i);
        if l >= 0, a = A(:, 1+l:end); bb = B(:, 1:end-l);
        else,      a = A(:, 1:end+l); bb = B(:, 1-l:end); end
        cc(i) = sum(a(:) .* bb(:)) / sqrt(sum(a(:).^2) * sum(bb(:).^2));
    end
    [~, im] = max(cc);
    lag = lags(im);
    if im > 1 && im < numel(lags)
        lag = lag + 0.5 * (cc(im-1) - cc(im+1)) / (cc(im-1) - 2*cc(im) + cc(im+1));
    end
    b = struct('lag', lag, 'cc', cc, 'lags', lags);
end

function s = sharpness(img)
    img = double(img) / mean(double(img(:)));
    [gx, gy] = imgradientxy(img, 'central');
    s = mean(sqrt(gx(:).^2 + gy(:).^2));
end

function [dy, dx] = phaseShift(ref, img)
    % shift of img content relative to ref (px), phase correlation + parabola
    [h, w] = size(ref);
    tap = (0.5 - 0.5 * cos(2*pi*(0:h-1)'/(h-1))) * (0.5 - 0.5 * cos(2*pi*(0:w-1)/(w-1)));
    a = (ref - mean(ref(:))) .* tap;
    b = (img - mean(img(:))) .* tap;
    R = fft2(b) .* conj(fft2(a));
    R = R ./ max(abs(R), eps);
    cc = fftshift(real(ifft2(R)));
    [~, i] = max(cc(:));
    [py, px] = ind2sub(size(cc), i);
    dy = py - (floor(h/2) + 1);
    dx = px - (floor(w/2) + 1);
    if py > 1 && py < h
        dy = dy + 0.5 * (cc(py-1, px) - cc(py+1, px)) / (cc(py-1, px) - 2*cc(py, px) + cc(py+1, px));
    end
    if px > 1 && px < w
        dx = dx + 0.5 * (cc(py, px-1) - cc(py, px+1)) / (cc(py, px-1) - 2*cc(py, px) + cc(py, px+1));
    end
end

function segLines(breaks)
    for b = reshape(breaks, 1, [])
        xline(b, '--', 'Color', [.5 .5 .5]);
    end
end

function v = vec(x)
    v = double(x(:));
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end

function S = mergeStruct(S, T)
    for f = fieldnames(T)'
        S.(f{1}) = T.(f{1});
    end
end

function m = bwr()
    n = 128;
    m = [linspace(0, 1, n)' linspace(0, 1, n)' ones(n, 1); ones(n, 1) linspace(1, 0, n)' linspace(1, 0, n)'];
end
