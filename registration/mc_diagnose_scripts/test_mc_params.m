% test_mc_params.m
%
% A/B test of motion-correction settings on the first nFrames frames of a
% recording, to see which setting gives the sharpest mean image before
% re-running a whole session.
%
% Each variant runs sbx_motion_correct on the same frames (same channel,
% crop, dropFirstFrame) with its own option set; the mean image of the
% registered movie is then compared on the SAME pixels (intersection of the
% variants' output rectangles). The registered tif of each variant is
% deleted afterwards unless keepTifs = true.
%
% Compared per variant:
%   sharpness    mean gradient magnitude of the mean image / its mean
%                (higher = sharper; compare only between variants)
%   lapVar       variance of the Laplacian, same normalisation
%   zoomSharp    sharpness inside the zoom patch (brightest region)
%   shifts       rigid shift SD, fraction of outlier frames, median cmax
%   NR           mean / p99 / max block shift (px) and fraction of
%                low-SNR block estimates, for non-rigid variants
%   time         seconds
%
% Edit the 'variants' list below. Each variant is name + options that are
% passed on top of baseOpts (later wins). Options: see help sbx_motion_correct.
%
% Output (in outDir): mc_param_test_means.png, mc_param_test_zoom.png,
% mc_param_test_summary.csv, mc_param_test_results.mat

%%%%%%%%%%%%%%%%%%
%%% USER INPUT %%%
%%%%%%%%%%%%%%%%%%
clear; close all
% .sbx recording(s): one path, or the segments of one recording, in order.
% The first nFrames frames (after dropping the first frame of each file)
% are used.
sbxFiles = {'Y:\mouse1\261009\ACQ1_000_001'};
nFrames  = 5000;
channel  = 1;                 % PMT channel to correct and compare (1 = green)
crop     = [1 80 Inf Inf];    % same as cfg.crop in batch_2p_Calcium
outDir   = '';                % '' = <first sbx folder>\mc_param_test
keepTifs = false;             % keep the registered 5k-frame tifs (~3-4 GB each)
overwrite = false;            % false = reuse a variant's saved result if it exists
zoomSize = 120;               % px, side of the zoomed patch

baseOpts = {'maxShift', 0.1, 'bidiShift', 0};  % applied to every variant (bidi off unless a variant sets it)

variants = struct('name', {}, 'opts', {});
% The diagnostics showed a +2.7 px even/odd line offset (bidirectional scan
% phase) that all earlier registration variants left in place. A* = without
% the correction (comparators, reused from saved results when overwrite =
% false and the same sbxFiles/nFrames: set overwrite = true or a new outDir if
% you changed the recording); D* = with bidiShift 'auto'.
variants(end+1) = struct('name', 'A0_baseline_NR_subpix',  'opts', {{'nonrigid', true,  'subpixel', true}});   % current settings, no bidi correction
variants(end+1) = struct('name', 'A1_rigid_subpix',        'opts', {{'nonrigid', false, 'subpixel', true}});   % no bidi correction, rigid only
variants(end+1) = struct('name', 'D1_bidi_NR_subpix',      'opts', {{'bidiShift', 'auto', 'nonrigid', true,  'subpixel', true}});   % does the bidi fix sharpen the mean image?
variants(end+1) = struct('name', 'D2_bidi_rigid_subpix',   'opts', {{'bidiShift', 'auto', 'nonrigid', false, 'subpixel', true}});   % with the comb gone, is non-rigid worth it?
variants(end+1) = struct('name', 'D3_bidi_NR_strip64',     'opts', {{'bidiShift', 'auto', 'nonrigid', true,  'blockSize', [64 4096]}});  % full-width row strips
%%%%%%%%%%%%%%%%%%%%%%%
%%% END: USER INPUT %%%
%%%%%%%%%%%%%%%%%%%%%%%

here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
addpath(genpath(fileparts(here)));     % parent folder: sbx_motion_correct, sbx_method
sbxFiles = cellstr(sbxFiles);
if isempty(outDir), outDir = fullfile(fileparts(sbxFiles{1}), 'mc_param_test'); end
if ~isfolder(outDir), mkdir(outDir); end
if isempty(gcp('nocreate')) && license('test', 'Distrib_Computing_Toolbox')
    try, parpool('Threads'); catch, end
end

nV = numel(variants);
R = cell(nV, 1);
for v = 1:nV
    name = variants(v).name;
    resFile = fullfile(outDir, [name '_result.mat']);
    if ~overwrite && isfile(resFile)
        fprintf('[%d/%d] %s: reusing %s\n', v, nV, name, resFile);
        R{v} = load(resFile);
        continue
    end
    fprintf('[%d/%d] %s ...\n', v, nV, name);
    opts = mergeOpts(baseOpts, variants(v).opts);
    tif = fullfile(outDir, [name '.tif']);
    t0 = tic;
    out = sbx_motion_correct(sbxFiles, 'outfile', tif, 'channel', channel, 'dropFirstFrame', true, ...
        'crop', crop, 'maxFrames', nFrames, 'verbose', false, opts{:});
    r = struct('name', name, 'opts', {opts}, 'time', toc(t0), 'meanImg', out.meanImg, ...
        'cropOut', out.crop.out, 'dy', out.dy, 'dx', out.dx, 'bad', out.bad, 'cmax', out.cmax, ...
        'nr', nrStats(out.nonrigid), 'bidiLag', out.bidi.lag, ...
        'bidiApplied', out.bidi.applied(2) - out.bidi.applied(1), 'bidiResid', out.bidi.lagAfter);
    save(resFile, '-struct', 'r');
    R{v} = r;
    if ~keepTifs
        delete(tif);
    end
end

%% compare on common pixels
O = cell2mat(cellfun(@(r) double(r.cropOut(:)'), R, 'UniformOutput', false));   % [r1 c1 r2 c2]
common = [max(O(:, 1)), max(O(:, 2)), min(O(:, 3)), min(O(:, 4))];
fprintf('\ncommon output rectangle (raw px): [%s], %d x %d\n', num2str(common), ...
    common(3) - common(1) + 1, common(4) - common(2) + 1);
imgs = cell(nV, 1);
for v = 1:nV
    o = O(v, :);
    imgs{v} = double(R{v}.meanImg(common(1)-o(1)+1 : common(3)-o(1)+1, common(2)-o(2)+1 : common(4)-o(2)+1));
end

% zoom patch: brightest region of the baseline (smoothed) mean image
sm = imgaussfilt(imgs{1}, 15);
[~, im] = max(sm(:));
[py, px] = ind2sub(size(sm), im);
h2 = floor(zoomSize / 2);
zr = max(1, min(size(sm, 1) - zoomSize + 1, py - h2)) + (0:zoomSize-1);
zc = max(1, min(size(sm, 2) - zoomSize + 1, px - h2)) + (0:zoomSize-1);

T = table('Size', [nV 15], 'VariableTypes', [{'string'}, repmat({'double'}, 1, 14)], 'VariableNames', ...
    {'variant', 'sharpness', 'lapVar', 'zoomSharp', 'dySD', 'dxSD', 'badFrac', 'medCmax', ...
     'nrMean', 'nrP99', 'nrMax', 'timeS', 'bidiLag', 'bidiApplied', 'bidiResid'});
for v = 1:nV
    r = R{v};
    I = imgs{v} / mean(imgs{v}(:));
    Iz = imgs{v}(zr, zc); Iz = Iz / mean(Iz(:));
    nr = r.nr;
    T(v, :) = {string(r.name), grad(I), var(del2(I), 0, 'all'), grad(Iz), std(r.dy), std(r.dx), ...
        mean(r.bad), median(r.cmax), nr.mean, nr.p99, nr.max, r.time, ...
        fieldOr(r, 'bidiLag'), fieldOr(r, 'bidiApplied'), fieldOr(r, 'bidiResid')};   % NaN in results saved before bidiShift existed
end
T.sharpRel = T.sharpness / T.sharpness(1);
T.zoomRel  = T.zoomSharp / T.zoomSharp(1);
disp(T);
writetable(T, fullfile(outDir, 'mc_param_test_summary.csv'));

%% figures (same contrast for every variant)
lim = prctile(cell2mat(cellfun(@(x) x(:), imgs, 'UniformOutput', false)), [1 99.7]);
nc = ceil(sqrt(nV));
nrw = ceil(nV / nc);
f1 = figure('Position', [50 50 1500 900]);
tiledlayout(nrw, nc, 'TileSpacing', 'compact');
for v = 1:nV
    nexttile; imagesc(imgs{v}, lim); axis image off; colormap(gca, gray);
    title(sprintf('%s\nsharp %.3f (x%.2f)', strrep(R{v}.name, '_', '\_'), T.sharpness(v), T.sharpRel(v)), 'FontSize', 8);
    rectangle('Position', [zc(1) zr(1) zoomSize zoomSize], 'EdgeColor', 'y');
end
saveas(f1, fullfile(outDir, 'mc_param_test_means.png'));
f2 = figure('Position', [50 50 1500 900]);
tiledlayout(nrw, nc, 'TileSpacing', 'compact');
for v = 1:nV
    nexttile; imagesc(imgs{v}(zr, zc), lim); axis image off; colormap(gca, gray);
    title(sprintf('%s\nzoom sharp %.3f (x%.2f)', strrep(R{v}.name, '_', '\_'), T.zoomSharp(v), T.zoomRel(v)), 'FontSize', 8);
end
saveas(f2, fullfile(outDir, 'mc_param_test_zoom.png'));
save(fullfile(outDir, 'mc_param_test_results.mat'), 'T', 'imgs', 'common', 'zr', 'zc', 'variants');
fprintf('Saved in %s\n', outDir);
fprintf(['Reading it: sharpRel > 1 = sharper than the baseline. If integer-shift variants win, bilinear\n' ...
    'resampling is the blur; if rigid beats NR, the non-rigid warp adds noise; if B1/B2 win, the\n' ...
    'large block shifts were noise. A/B differences < ~2-3 %% are within noise.\n']);


%% Local functions
function s = grad(I)
    [gx, gy] = imgradientxy(I, 'central');
    s = mean(sqrt(gx(:).^2 + gy(:).^2));
end

function v = fieldOr(s, f)
    v = NaN;
    if isfield(s, f)
        v = s.(f);
    end
end

function s = nrStats(nr)
    s = struct('mean', nan, 'p99', nan, 'max', nan, 'lowFrac', nan);
    if isempty(nr), return, end
    L = double(hypot(nr.dy, nr.dx));
    s.mean = mean(L, 'all');
    s.p99 = prctile(L(:), 99);
    s.max = max(L, [], 'all');
    if isfield(nr, 'lowSNR') && ~isempty(nr.lowSNR), s.lowFrac = mean(nr.lowSNR, 'all'); end
end

function o = mergeOpts(base, over)
    % name-value cells; entries of 'over' replace the same names in 'base'
    o = base(:)';
    for i = 1:2:numel(over)
        k = find(strcmpi(o(1:2:end), over{i}), 1);
        if isempty(k)
            o(end+1:end+2) = over(i:i+1); %#ok<AGROW>
        else
            o{2*k} = over{i+1};
        end
    end
end
