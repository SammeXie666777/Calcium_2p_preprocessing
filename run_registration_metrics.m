% DEMO_REGISTRATION_METRICS  Check and report motion-correction quality.
%
% Runs registration_metrics.m on a motion-corrected movie (and, if given,
% the raw movie before correction), prints a report in the command window
% and saves all figures. Two modes:
%   'synthetic' - builds a movie of sparse "neurons"; the clean version is
%                 treated as the corrected movie and a copy with slow rigid
%                 drift + non-rigid wobble as the raw movie. Use it to see
%                 what good vs bad looks like.
%   'data'      - loads ConcatenatedMovie_MC.tif (+ its _shifts.mat from
%                 sbx_motion_correct) and optionally a raw tif.
%
% Report sections:
%   1. Shifts applied by motion correction (data mode, from _shifts.mat)
%   2. PC metric (Suite2p): shift between top/bottom PC averages, per PC
%   3. Frame-to-mean correlation: per-frame similarity to the mean image
%   4. Kymograph wiggle: how far edges along each kymograph line move
%      between time bins (1-D cross-correlation with the mean profile)
% Each check gets OK / CHECK against the rule-of-thumb thresholds in thr
% (heuristics for this lab's data, not Suite2p or CaImAn standards).
%
% Requires registration_metrics.m (this folder).

%%%%%%%%%%%%%%%%%%
%%% USER INPUT %%%
%%%%%%%%%%%%%%%%%%
clear; close all
mode = 'data';          % 'synthetic' or 'data'

% data mode
mcTif  = 'Y:\mouse1\260928\932_3dir_rec000\ConcatenatedMovie_MC.tif';                 % ...\<session>\ConcatenatedMovie_MC.tif
rawTif = 'Y:\mouse1\260928\acq1_000_001_ch1.tif';                 % optional raw tif of the same recording ('' = none),
                             % exported with the same 'crop' as motion correction;
                             % cropped to the MC output with outRows/outCols
maxFrames = Inf;             % use only the first N frames (speed); the movies are
                             % streamed from disk, so memory does not grow with N
margin = 0;                  % extra px ignored on every side of the MC movie

runPCraw = true;             % also run the PC metric on the raw movie (slower)
saveFigs = true;
outDir   = fullfile(pwd, 'reg_metrics_figs');

% rule-of-thumb thresholds for OK / CHECK
thr.pcTotal  = 0.5;          % px, max over PCs of mean(rigid+NR)
thr.corrDip  = 0.01;         % fraction of frames dipping below median - 5 robust SD
thr.kymoSD   = 0.5;          % px, SD of the edge position along a kymograph line
thr.badFrac  = 0.01;         % fraction of frames flagged as outliers during MC
%%%%%%%%%%%%%%%%%%%%%%%
%%% END: USER INPUT %%%
%%%%%%%%%%%%%%%%%%%%%%%

figsBefore = findall(groot, 'Type', 'figure');

%% LOAD OR BUILD MOVIES
S = [];
truth = '';
switch mode
    case 'synthetic'
        [mc, raw, truth] = makeSynthetic();
        name = 'synthetic';
        opts = {'yrange', [8 size(mc,1)-8], 'xrange', [8 size(mc,2)-8], 'blockSize', [96 96]};
    case 'data'
        if ~isfile(mcTif)
            error('mcTif not found: %s', mcTif);
        end
        [d, n] = fileparts(mcTif);
        name = n;
        mc = mcTif;                  % path: registration_metrics streams it from disk
        sf = fullfile(d, [n '_shifts.mat']);
        if isfile(sf)
            S = load(sf, 'dy', 'dx', 'bad', 'cmax', 'nonrigid', 'outRows', 'outCols');
        else
            fprintf('  no %s; skipping the applied-shifts section\n', sf);
        end
        mcSize = tifFrameSize(mcTif);
        raw = [];
        if ~isempty(rawTif)
            raw = rawSource(rawTif, mcSize, S);
        end
        opts = {'yrange', [1+margin mcSize(1)-margin], 'xrange', [1+margin mcSize(2)-margin]};
    otherwise
        error('mode must be ''synthetic'' or ''data''.');
end

%% METRICS
fprintf('Running registration_metrics on the corrected movie ...\n');
M = registration_metrics(mc, opts{:}, 'rawMov', raw, 'rawPC', runPCraw, ...
    'maxFrames', maxFrames, 'plot', true);
Mr = [];
if ~isempty(M.regDXRaw)
    Mr = struct('regDX', M.regDXRaw, 'topMean', M.topMeanRaw, 'botMean', M.botMeanRaw);
end
W = struct('H', kymoWiggle(M.kymo.H), 'V', kymoWiggle(M.kymo.V), 'Hraw', [], 'Vraw', []);
if ~isempty(M.kymo.Hraw)
    W.Hraw = kymoWiggle(M.kymo.Hraw);
    W.Vraw = kymoWiggle(M.kymo.Vraw);
end

%% REPORT
lbl = {'rigid', 'mean NR', 'max NR', 'mean(rigid+NR)'};
flags = {};
sep = repmat('=', 1, 72);
fprintf('\n%s\nMOTION CORRECTION REPORT: %s\n%s\n', sep, name, sep);
fprintf('Corrected movie: %d x %d px, %d frames', M.movSize);
if ~isempty(M.rawSize)
    fprintf(' | raw movie: %d frames', M.rawSize(3));
end
fprintf('\nAnalysed region: rows %d-%d, cols %d-%d\n', M.opts.yrange, M.opts.xrange);
if ~isempty(truth)
    fprintf('Injected motion in the raw movie: %s\n', truth);
end

% 1. applied shifts
fprintf('\n[1] Shifts applied by motion correction\n');
if isempty(S)
    fprintf('    (no _shifts.mat: synthetic mode, or not found next to the tif)\n');
else
    mag = hypot(S.dy, S.dx);
    fprintf('    rigid dy: %+.2f to %+.2f px (SD %.2f) | dx: %+.2f to %+.2f px (SD %.2f)\n', ...
        min(S.dy), max(S.dy), std(S.dy), min(S.dx), max(S.dx), std(S.dx));
    fprintf('    |shift|: median %.2f px, 99th pct %.2f px, max %.2f px\n', ...
        median(mag), pct(mag, 99), max(mag));
    badFrac = mean(S.bad);
    fprintf('    outlier frames (shift replaced by moving median): %d (%.2f%%)  %s\n', ...
        sum(S.bad), 100*badFrac, okStr(badFrac <= thr.badFrac));
    if badFrac > thr.badFrac, flags{end+1} = 'many outlier frames during MC'; end
    if isfield(S, 'nonrigid') && isstruct(S.nonrigid) && ~isempty(S.nonrigid)
        nrMag = hypot(double(S.nonrigid.dy), double(S.nonrigid.dx));
        fprintf('    non-rigid block shifts: median %.2f px, 99th pct %.2f px (limit %d px); low-SNR blocks %.1f%%\n', ...
            median(nrMag(:)), pct(nrMag(:), 99), S.nonrigid.maxShift, 100*mean(S.nonrigid.lowSNR(:)));
    end
end

% 2. PC metric
fprintf('\n[2] PC metric (shift between top/bottom PC averages; ideally ~0 px)\n');
fprintf('    %-16s %12s %12s', '', 'corrected', 'at PC #');
if ~isempty(Mr), fprintf(' %12s', 'raw'); end
fprintf('\n');
[mx, iPC] = max(M.regDX, [], 1);
for c = 1:4
    fprintf('    %-16s %10.3f px %12d', lbl{c}, mx(c), iPC(c));
    if ~isempty(Mr), fprintf(' %9.3f px', max(Mr.regDX(:, c))); end
    fprintf('\n');
end
fprintf('    -> max total residual %.3f px  %s\n', mx(4), okStr(mx(4) <= thr.pcTotal));
if mx(4) > thr.pcTotal
    flags{end+1} = sprintf('PC metric: PC %d has %.2f px residual shift (see its top - bottom image)', iPC(4), mx(4));
end

% 3. frame-to-mean correlation
fprintf('\n[3] Frame-to-mean correlation (higher and flatter = better)\n');
cs = corrStats(M.frameCorr);
fprintf('    corrected: median %.3f | 1st pct %.3f | min %.3f | dips %d frames (%.2f%%)  %s\n', ...
    cs.med, cs.p1, cs.min, cs.nDip, 100*cs.dipFrac, okStr(cs.dipFrac <= thr.corrDip));
if ~isempty(M.frameCorrRaw)
    rs = corrStats(M.frameCorrRaw);
    fprintf('    raw:       median %.3f | 1st pct %.3f | min %.3f\n', rs.med, rs.p1, rs.min);
    fprintf('    change:    median %+.3f, 1st pct %+.3f\n', cs.med - rs.med, cs.p1 - rs.p1);
end
if ~isempty(cs.dipFrames)
    fprintf('    worst corrected frames: %s\n', mat2str(cs.dipFrames(1:min(10, end))'));
end
if cs.dipFrac > thr.corrDip
    flags{end+1} = sprintf('%.2f%% of frames dip in correlation (check them against behavior)', 100*cs.dipFrac);
end

% 4. kymograph wiggle
fprintf('\n[4] Kymograph wiggle (edge movement along each line, %d-frame bins)\n', M.kymo.bin);
fprintf('    lines: horizontal at row %d, vertical at col %d (of the analysed region)\n', M.kymo.row, M.kymo.col);
names = {'horizontal (x)', 'vertical (y)'};
fld = {'H', 'V'};
for k = 1:2
    w = W.(fld{k});
    fprintf('    %-15s corrected: SD %.2f px, max |dev| %.2f px', names{k}, std(w), max(abs(w - median(w))));
    if ~isempty(W.([fld{k} 'raw']))
        wr = W.([fld{k} 'raw']);
        fprintf(' | raw: SD %.2f px, max |dev| %.2f px', std(wr), max(abs(wr - median(wr))));
    end
    fprintf('  %s\n', okStr(std(w) <= thr.kymoSD));
    if std(w) > thr.kymoSD
        flags{end+1} = sprintf('kymograph %s line wiggles (SD %.2f px)', names{k}, std(w)); %#ok<SAGROW>
    end
end

% summary
fprintf('\n%s\n', repmat('-', 1, 72));
if isempty(flags)
    fprintf('SUMMARY: all checks OK. No sustained residual motion detected.\n');
else
    fprintf('SUMMARY: %d check(s) to look at:\n', numel(flags));
    fprintf('  - %s\n', flags{:});
end
fprintf(['Note: none of these measure z-motion directly. Look for cells fading in and out\n' ...
         'in the kymographs and the PC top - bottom images.\n%s\n\n'], sep);

%% SUMMARY FIGURE
figure('Name', 'Motion correction summary', 'Color', 'w', 'Position', [50 50 1500 750]);
subplot(2,3,1);
vals = mx(:);
leg = {'corrected'};
if ~isempty(Mr)
    vals = [max(Mr.regDX, [], 1)' vals];
    leg = {'raw', 'corrected'};
end
bar(vals); box off
set(gca, 'XTickLabel', lbl); ylabel('max over PCs (px)');
yline(thr.pcTotal, '--', 'threshold'); legend(leg, 'Location', 'northwest'); legend boxoff
title('PC metric');

subplot(2,3,2); hold on
if ~isempty(Mr), plot(Mr.regDX(:, 4), '.-', 'Color', [0.85 0.33 0.10], 'MarkerSize', 12); end
plot(M.regDX(:, 4), 'k.-', 'MarkerSize', 12);
yline(thr.pcTotal, '--'); box off
xlabel('PC #'); ylabel('mean(rigid+NR) (px)'); title('residual shift per PC');
legend(leg, 'Location', 'northeast'); legend boxoff

subplot(2,3,3);
if ~isempty(S)
    plot(S.dy, 'b'); hold on; plot(S.dx, 'r');
    plot(find(S.bad), S.dy(S.bad), 'kx');
    legend({'dy', 'dx', 'outlier'}, 'Location', 'best'); legend boxoff
    title('applied rigid shifts');
else
    text(0.5, 0.5, 'no _shifts.mat', 'HorizontalAlignment', 'center', 'Interpreter', 'none');
    title('applied rigid shifts'); axis off
end
xlabel('frame'); ylabel('px'); box off

subplot(2,3,4);
showPCdiff(M, 'corrected: PC1 top - bottom');
subplot(2,3,5);
if ~isempty(Mr)
    showPCdiff(Mr, 'raw: PC1 top - bottom');
else
    axis off; text(0.5, 0.5, 'no raw movie', 'HorizontalAlignment', 'center');
end

subplot(2,3,6); hold on
if ~isempty(W.Hraw)
    plot(M.kymo.tRaw, W.Hraw - median(W.Hraw), 'Color', [1 0.6 0.4]);
    plot(M.kymo.tRaw, W.Vraw - median(W.Vraw), 'Color', [0.6 0.6 1]);
end
plot(M.kymo.t, W.H - median(W.H), 'Color', [0.85 0.33 0.10]);
plot(M.kymo.t, W.V - median(W.V), 'Color', [0 0.2 0.8]);
box off; xlabel('frame'); ylabel('edge position (px)');
if ~isempty(W.Hraw)
    legend({'raw horizontal', 'raw vertical', 'corrected horizontal', 'corrected vertical'}, 'Location', 'best');
else
    legend({'horizontal', 'vertical'}, 'Location', 'best');
end
legend boxoff
title('kymograph wiggle');
sgtitle(sprintf('%s: motion correction summary', name), 'Interpreter', 'none');
drawnow;

%% SAVE FIGURES
if saveFigs
    if ~isfolder(outDir), mkdir(outDir); end
    figs = findall(groot, 'Type', 'figure');
    figs = figs(~arrayfun(@(f) any(f == figsBefore), figs));   % only figures made by this run
    for f = figs(:)'
        fn = regexprep([name '_' f.Name], '[^\w-]+', '_');
        exportgraphics(f, fullfile(outDir, [fn '.png']), 'Resolution', 150);
    end
    fprintf('Saved %d figure(s) to %s\n', numel(figs), outDir);
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% LOCAL FUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%

function [good, bad, truth] = makeSynthetic()
% Sparse "neurons" on a neuropil-like texture. good = no motion (stands in
% for the corrected movie); bad = same movie with slow rigid drift and a
% non-rigid gradient (stands in for the raw movie).
rng(1);
Ly = 256; Lx = 256; T = 3000; nCell = 120;
[yy, xx] = ndgrid(1:Ly, 1:Lx);
cy = 10 + rand(nCell,1)*(Ly-20); cx = 10 + rand(nCell,1)*(Lx-20);
A = zeros(Ly*Lx, nCell, 'single');
for i = 1:nCell
    A(:,i) = reshape(exp(-((yy-cy(i)).^2 + (xx-cx(i)).^2) / (2*3^2)), [], 1);
end
spk = single(rand(nCell, T) < 0.01);
act = filter(1, [1 -0.9], spk, [], 2);
bg  = imgaussfilt_nt(rand(Ly, Lx), 8) * 2;          % neuropil-like texture
base = reshape(sum(A, 2), Ly, Lx) + bg;

drift = 2.0 * sin(2*pi*(1:T)/800);                  % rigid, px
nrAmp = 1.0 * sin(2*pi*(1:T)/350);                  % non-rigid, px
good = zeros(Ly, Lx, T, 'single');
bad  = zeros(Ly, Lx, T, 'single');
for t = 1:T
    F = base + reshape(A * act(:,t), Ly, Lx);
    good(:,:,t) = F + 0.3*randn(Ly, Lx);
    dy = drift(t) + nrAmp(t) * (xx - Lx/2) / (Lx/2);   % y-shift varies across x
    dx = 0.5*drift(t) * ones(Ly, Lx);
    bad(:,:,t) = interp2(F, xx - dx, yy - dy, 'linear', 0) + 0.3*randn(Ly, Lx);
end
truth = 'rigid dy +/-2 px, dx +/-1 px (800-frame period); non-rigid dy +/-1 px across x (350-frame period)';
end

function sz = tifFrameSize(path)
% [rows cols] of the first page of a tif, without reading image data.
w = warning('off', 'imageio:tiffmexutils:libtiffWarning');
restoreW = onCleanup(@() warning(w));
t = Tiff(path, 'r');
closeT = onCleanup(@() t.close());
sz = [t.getTag('ImageLength') t.getTag('ImageWidth')];
end

function raw = rawSource(rawTif, mcSize, S)
% Raw movie for registration_metrics, covering the pixels of the MC output:
% the path itself if the sizes match, else struct('path', 'rows', 'cols')
% with outRows/outCols of the _shifts.mat (which refer to the raw frame
% after the motion-correction 'crop').
rawSize = tifFrameSize(rawTif);
if isequal(rawSize, mcSize)
    raw = rawTif;
    return
end
if isempty(S) || ~isfield(S, 'outRows') || max(S.outRows) > rawSize(1) || max(S.outCols) > rawSize(2)
    error(['Raw movie is %dx%d px and MC movie %dx%d px, and the _shifts.mat outRows/outCols ' ...
        'do not fit. Export the raw tif with the same ''crop'' as motion correction.'], ...
        rawSize, mcSize);
end
raw = struct('path', rawTif, 'rows', S.outRows(:)', 'cols', S.outCols(:)');
end

function d = kymoWiggle(P, maxLag)
% Position of the edges along a kymograph line in each time bin, relative
% to the mean profile: lag (px, subpixel) maximizing the correlation of the
% bin's profile with the mean profile. Positive = content moved toward
% higher row/column. Slow brightness along the line is removed first so
% that edges dominate.
if nargin < 2, maxLag = 10; end
P = double(P);
P = P - movmean(P, 15, 1);
ref = mean(P, 2);
L = size(P, 1);
maxLag = min(maxLag, floor(L/4));
lags = -maxLag:maxLag;
C = zeros(numel(lags), size(P, 2));
for k = 1:numel(lags)
    l = lags(k);
    i = max(1, 1-l):min(L, L-l);                  % ref(i) vs P(i+l)
    a = ref(i) - mean(ref(i));
    B = P(i+l, :) - mean(P(i+l, :), 1);
    C(k, :) = (a' * B) ./ (norm(a) * sqrt(sum(B.^2, 1)) + eps);
end
[~, im] = max(C, [], 1);
d = lags(im);
inner = find(im > 1 & im < numel(lags));          % 3-point parabolic subpixel
idx = sub2ind(size(C), im(inner), inner);
cm = C(idx - 1); c0 = C(idx); cp = C(idx + 1);
den = cm - 2*c0 + cp;
s = zeros(size(c0));
ok = den < 0;
s(ok) = 0.5 * (cm(ok) - cp(ok)) ./ den(ok);
d(inner) = d(inner) + max(min(s, 0.5), -0.5);
d = d(:);
end

function s = corrStats(r)
% Summary of a frame-to-mean correlation trace. Dips: frames below
% median - 5 robust SD (1.4826 * MAD).
frames = find(~isnan(r));
r = r(frames);
s.med = median(r);
s.p1 = pct(r, 1);
s.min = min(r);
thrDip = s.med - 5 * 1.4826 * median(abs(r - s.med));
isDip = r < thrDip;
s.nDip = sum(isDip);
s.dipFrac = mean(isDip);
[~, o] = sort(r(isDip));
dips = frames(isDip);
s.dipFrames = dips(o);
end

function showPCdiff(M, ttl)
d = M.topMean(:,:,1) - M.botMean(:,:,1);
lim = max(abs(d(:)));
if lim == 0, lim = 1; end
imagesc(d, [-lim lim]); axis image off; colormap(gca, redblue); colorbar
title(sprintf('%s\n(rigid %.2f px, total %.2f px)', ttl, M.regDX(1, 1), M.regDX(1, 4)));
end

function v = pct(x, q)
% Percentile without the Statistics Toolbox.
x = sort(double(x(:)));
if isscalar(x)
    v = x;
else
    v = interp1(linspace(0, 100, numel(x)), x, q);
end
end

function s = okStr(tf)
if tf
    s = 'OK';
else
    s = 'CHECK';
end
end

function y = imgaussfilt_nt(x, s)   % toolbox-free Gaussian blur
k = exp(-(-3*s:3*s).^2 / (2*s^2)); k = k / sum(k);
y = conv2(k, k, x, 'same');
end

function c = redblue
n = 128; a = linspace(0,1,n)';
c = [[a a ones(n,1)]; [ones(n,1) flipud(a) flipud(a)]];
end
